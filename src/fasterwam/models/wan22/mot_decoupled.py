"""Run final-layer readout and the shared per-head video-layer mixer.

Video and action experts may have different depths. The RoPE-aware subclass
reuses the same parameter layout for the published layer-fusion models."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from .mot import MoT
from fasterwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class PreparedActionVideoKV(list[dict[str, torch.Tensor]]):
    """Per-action-layer video KV cache prepared once for denoising.

    The list subclass deliberately preserves the historical cache API while
    marking that source selection / learned layer fusion has already run.  A
    denoising loop can therefore share this object across every Euler step
    without repeating static video-KV work.
    """

# Expert key constants used to index into the mixtures dict.
VIDEO_EXPERT_KEY = "video"
ACTION_EXPERT_KEY = "action"


def compute_kv_source_mapping(
    mode: str,
    video_num_layers: int,
    action_num_layers: int,
) -> list[int]:
    """Resolve final-layer readout, all-layer fusion, or pretrained initialization.

    ``uniform_end`` selects source layers when initializing a shallower action
    backbone from pretrained weights; it is not an additional public readout."""
    N, M = video_num_layers, action_num_layers
    if mode == "final_only":
        mapping = [N - 1] * M
    elif mode == "uniform_end":
        if M > N:
            raise ValueError(
                f"uniform_end requires action_num_layers <= video_num_layers, "
                f"got M={M} > N={N}"
            )
        mapping = [round((k + 1) * N / M) - 1 for k in range(M)]
    elif mode in {"new_fused_kv"}:
        mapping = list(range(N))
    else:
        raise ValueError(
            f"Unknown kv_source mode: {mode!r}. "
            "Expected 'final_only', 'uniform_end', or 'new_fused_kv'."
        )
    if mode not in {"new_fused_kv"}:
        assert len(mapping) == M
    else:
        assert len(mapping) > 0

    assert all(0 <= v < N for v in mapping), (
        f"Mapping values out of range [0, {N}): {mapping}"
    )
    return mapping


class MoTDecoupled(MoT):
    """Video/action experts with different depths and first-frame action visibility.

    The base forward reads selected video-layer K/V. The RoPE-aware subclass
    implements learned per-head fusion through the shared mixer and cache helpers."""

    enable_new_fused_kv_key_norm: bool = False
    # ``full`` is the inert historical metadata value for final-layer readout.
    supported_new_fused_kv_projection_modes = {"full", "simple_head_softmax"}
    supported_new_fused_kv_simple_head_softmax_fuse_modes = {"all"}

    def __init__(
        self,
        mixtures: dict[str, nn.Module],
        video_num_layers: int,
        action_num_layers: int,
        num_heads: int,
        attn_head_dim: int,
        mot_checkpoint_mixed_attn: bool = True,
        kv_source_mapping: list[int] | None = None,
        kv_source_mode: str = "final_only",
        new_fused_kv_projection_mode: str = "full",
        new_fused_kv_simple_head_softmax_fuse_mode: str = "all",
    ) -> None:
        # Bypass MoT.__init__ to avoid the equal-layer-count check (mot.py:39-41).
        # We call nn.Module.__init__ directly so that PyTorch module registration
        # (e.g., nn.ModuleDict) still works correctly.
        nn.Module.__init__(self)

        # ---- Validate mixtures dict ----
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if VIDEO_EXPERT_KEY not in mixtures or ACTION_EXPERT_KEY not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )

        # ---- Store as nn.ModuleDict (critical for parameter tracking) ----
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())

        # ---- Store layer counts per expert ----
        self.video_num_layers = video_num_layers
        self.action_num_layers = action_num_layers

        # Validate that the experts actually have the expected number of blocks.
        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        action_expert = self.mixtures[ACTION_EXPERT_KEY]

        if len(video_expert.blocks) != video_num_layers:
            raise ValueError(
                f"Video expert has {len(video_expert.blocks)} blocks but "
                f"video_num_layers={video_num_layers}"
            )
        if len(action_expert.blocks) != action_num_layers:
            raise ValueError(
                f"Action expert has {len(action_expert.blocks)} blocks but "
                f"action_num_layers={action_num_layers}"
            )

        # ---- Enforce matching head structure (hard constraint from concat SDPA) ----
        # See design doc Section 3.5: both experts must use the same num_heads
        # and attn_head_dim so that the single reshape in flash_attention does
        # not scramble head boundaries.
        for name in self.expert_order:
            expert = self.mixtures[name]
            if expert.num_heads != num_heads:
                raise ValueError(
                    f"Expert '{name}' has num_heads={expert.num_heads} but "
                    f"MoTDecoupled requires num_heads={num_heads}"
                )
            if expert.attn_head_dim != attn_head_dim:
                raise ValueError(
                    f"Expert '{name}' has attn_head_dim={expert.attn_head_dim} "
                    f"but MoTDecoupled requires attn_head_dim={attn_head_dim}"
                )

        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn

        if mot_checkpoint_mixed_attn:
            logger.info(
                "MoTDecoupled: Using gradient checkpointing for mixed attention."
            )

        # NOTE: We intentionally do NOT set self.num_layers. This prevents any
        # inherited method that references self.num_layers from silently
        # operating on a wrong layer count.

        supported_kv_source_modes = {"final_only", "new_fused_kv"}
        if kv_source_mode not in supported_kv_source_modes:
            raise ValueError(f"Unsupported kv_source_mode {kv_source_mode!r}")
        self.kv_source_mode = kv_source_mode
        expected_projection = "simple_head_softmax" if kv_source_mode == "new_fused_kv" else "full"
        if new_fused_kv_projection_mode != expected_projection:
            raise ValueError(
                f"{kv_source_mode} requires new_fused_kv_projection_mode={expected_projection!r}"
            )
        self.new_fused_kv_projection_mode = new_fused_kv_projection_mode
        if new_fused_kv_simple_head_softmax_fuse_mode != "all":
            raise ValueError("Published layer fusion requires fuse_mode='all'")
        self.new_fused_kv_simple_head_softmax_fuse_mode = "all"

        # ---- Guard: new_fused_kv requires the action-aligned 3D-RoPE subclass ----
        # new_fused_kv is DEFINED by the action-aligned 3D-RoPE path plus the
        # fused-key RMSNorm, both of which live only in the subclass that sets
        # enable_new_fused_kv_key_norm=True (MoTDecoupledActionAlignedVideoRoPE,
        # fasterwam_decoupled.py). The base class lacks that RoPE path and never
        # allocates k_fused_norm, so constructing base MoTDecoupled with
        # new_fused_kv would SILENTLY train a structurally different model. We key
        # the predicate on the class ATTRIBUTE (not the class name) so any future
        # subclass that opts in via enable_new_fused_kv_key_norm=True still works.
        if (
            kv_source_mode == "new_fused_kv"
            and type(self).enable_new_fused_kv_key_norm is False
        ):
            raise ValueError(
                "kv_source_mode='new_fused_kv' is only supported by the "
                "action-aligned fixed-RoPE subclass "
                "(MoTDecoupledActionAlignedVideoRoPE, i.e. enable "
                "enable_new_fused_kv_key_norm=True / set fixed_rope=true); the "
                "base MoTDecoupled lacks the action-aligned 3D-RoPE path and the "
                "fused-key norm that define new_fused_kv, so constructing it here "
                "would silently train a different model. Use the fixed-RoPE class "
                "(fixed_rope=true) or choose kv_source_mode='final_only' instead."
            )

        if kv_source_mapping is None:
            if kv_source_mode in {"new_fused_kv"}:
                kv_source_mapping = list(range(video_num_layers))
            else:
                kv_source_mapping = [video_num_layers - 1] * action_num_layers

        if (
            kv_source_mode not in {"new_fused_kv"}
            and len(kv_source_mapping) != action_num_layers
        ):
            raise ValueError(
                f"kv_source_mapping length {len(kv_source_mapping)} != "
                f"action_num_layers {action_num_layers}"
            )
        if kv_source_mode in {"new_fused_kv"} and len(kv_source_mapping) == 0:
            raise ValueError(f"{kv_source_mode} requires at least one kv_source_mapping entry")
        if any(v < 0 or v >= video_num_layers for v in kv_source_mapping):
            raise ValueError(
                f"kv_source_mapping values must be in [0, {video_num_layers}), "
                f"got {kv_source_mapping}"
            )
        if list(kv_source_mapping) != sorted(kv_source_mapping):
            raise ValueError(
                f"kv_source_mapping must be sorted ascending, "
                f"got {kv_source_mapping}"
            )
        self.kv_source_mapping: list[int] = list(kv_source_mapping)

        self._action_schedule: dict[int, list[int]] = {}
        self.register_buffer(
            "kv_source_indices", torch.tensor(self.kv_source_mapping, dtype=torch.long),
            persistent=False,
        )
        if kv_source_mode == "new_fused_kv":
            for video_idx in self.kv_source_mapping:
                self._action_schedule[video_idx] = list(range(action_num_layers))
            self.k_fused_norm = nn.ModuleList([
                self._make_attn_rms_norm(video_expert)
                for _ in range(action_num_layers)
            ])
            self.lite_kv_fusion_logits = nn.Parameter(torch.zeros(
                num_heads, len(self.kv_source_mapping), action_num_layers,
            ))
        else:
            for action_idx, video_idx in enumerate(self.kv_source_mapping):
                self._action_schedule.setdefault(video_idx, []).append(action_idx)
            self.k_fused_norm = None
            self.register_buffer("lite_kv_fusion_logits", None, persistent=False)

        logger.info(
            "Initialized MoTDecoupled: video_layers=%d action_layers=%d readout=%s mapping=%s",
            video_num_layers, action_num_layers, kv_source_mode, self.kv_source_mapping,
        )
        action_cross_attn = getattr(action_expert, "action_text_cross_attn", True)
        for name in self.expert_order:
            expert = self.mixtures[name]
            num_params = sum(p.numel() for p in expert.parameters())
            num_trainable = sum(p.numel() for p in expert.parameters() if p.requires_grad)
            extra = ""
            if name == ACTION_EXPERT_KEY:
                extra = f", text_cross_attn={'on' if action_cross_attn else 'off'}"
            logger.info(
                f"  Expert '{name}': total={num_params / 1e9:.2f} B, "
                f"trainable={num_trainable / 1e9:.2f} B{extra}"
            )
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"  MoTDecoupled total: {total_params / 1e9:.2f} B, "
            f"trainable: {trainable_params / 1e9:.2f} B"
        )

    def _make_attn_rms_norm(self, expert: nn.Module) -> nn.Module:
        """Create an attention-width RMSNorm matching the expert implementation."""
        norm_k = expert.blocks[0].self_attn.norm_k
        norm_cls = type(norm_k)
        eps = getattr(norm_k, "eps", 1e-6)
        norm = norm_cls(self.num_heads * self.attn_head_dim, eps=eps)
        weight = getattr(norm_k, "weight", None)
        if weight is not None:
            norm = norm.to(device=weight.device, dtype=weight.dtype)
        return norm

    # ------------------------------------------------------------------
    # Override inherited lock-step methods with NotImplementedError
    # ------------------------------------------------------------------

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> None:
        """Raises NotImplementedError -- use ``forward_decoupled`` for training
        or ``prefill_video_kv`` + ``forward_action_with_video_kv`` for
        inference."""
        raise NotImplementedError(
            "Use forward_decoupled() for training or "
            "prefill_video_kv + forward_action_with_video_kv for inference"
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> None:
        """Raises NotImplementedError -- use ``prefill_video_kv`` instead."""
        raise NotImplementedError("Use prefill_video_kv instead")

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> None:
        """Raises NotImplementedError -- use ``forward_action_with_video_kv``
        instead."""
        raise NotImplementedError("Use forward_action_with_video_kv instead")

    # ------------------------------------------------------------------
    # New decoupled methods
    # ------------------------------------------------------------------


    def forward_decoupled(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run video layers and execute each action layer at its selected video source.

        Action gradients flow into the selected video K/V without detaching them.
        The RoPE-aware subclass overrides this loop for all-layer fusion."""
        # ---- Input validation ----
        for key in (VIDEO_EXPERT_KEY, ACTION_EXPERT_KEY):
            if key not in embeds_all:
                raise ValueError(f"Missing '{key}' in embeds_all")
            if key not in freqs_all:
                raise ValueError(f"Missing '{key}' in freqs_all")
            if key not in t_mod_all:
                raise ValueError(f"Missing '{key}' in t_mod_all")
            if key not in attention_masks:
                raise ValueError(f"Missing '{key}' in attention_masks")

        video_mask = attention_masks[VIDEO_EXPERT_KEY]
        action_mask = attention_masks[ACTION_EXPERT_KEY]

        if video_mask.ndim != 2 or video_mask.shape[0] != video_mask.shape[1]:
            raise ValueError(
                f"Video attention mask must be square 2D, got shape "
                f"{tuple(video_mask.shape)}"
            )

        video_seq_len = embeds_all[VIDEO_EXPERT_KEY].shape[1]
        action_seq_len = embeds_all[ACTION_EXPERT_KEY].shape[1]

        if video_mask.shape[0] != video_seq_len:
            raise ValueError(
                f"Video mask seq length {video_mask.shape[0]} != "
                f"video token seq length {video_seq_len}"
            )

        # Action mask should be [Sa, Sv + Sa]
        expected_action_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
        if action_mask.shape != torch.Size(expected_action_mask_shape):
            raise ValueError(
                f"Action attention mask must have shape {expected_action_mask_shape}, "
                f"got {tuple(action_mask.shape)}"
            )

        # ---- Video forward and all-layer KV capture ----
        # ---- Dispatch: fused MLP path vs interleaved path ----
        if not self._action_schedule:
            raise RuntimeError(
                "_action_schedule is empty: "
                "no action layers would execute. This is a misconfiguration."
            )


        # ---- Single-pass interleaved forward ----
        # One loop over video layers. Action layers run inline when their
        # source video layer is reached, via self._action_schedule.
        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        x_video = embeds_all[VIDEO_EXPERT_KEY]
        x_action = embeds_all[ACTION_EXPERT_KEY]
        video_freqs = freqs_all[VIDEO_EXPERT_KEY]
        video_t_mod = t_mod_all[VIDEO_EXPERT_KEY]
        action_freqs = freqs_all[ACTION_EXPERT_KEY]
        action_t_mod = t_mod_all[ACTION_EXPERT_KEY]
        video_context = context_all.get(VIDEO_EXPERT_KEY)
        action_context = context_all.get(ACTION_EXPERT_KEY)
        video_kv_cache: list[dict[str, torch.Tensor]] = []

        for video_layer_idx in range(self.video_num_layers):
            video_block = video_expert.blocks[video_layer_idx]
            (
                q, k, v,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v,
                attention_mask=video_mask,
            )

            x_video = self._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context,
            )

            video_kv_cache.append({"k": k, "v": v})

        stacked_k, stacked_v = self._stack_video_kv(video_kv_cache)

        # ---- Action forward with learned all-layer KV mixing ----
        for action_layer_idx in range(self.action_num_layers):
            action_block = action_expert.blocks[action_layer_idx]
            (
                q_a, k_a, v_a,
                res_a,
                gate_a,
                sh_a, sc_a, g_a,
                gc_a,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            k_video, v_video = self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=action_layer_idx,
            )
            k_cat = torch.cat([k_video, k_a], dim=1)
            v_cat = torch.cat([v_video, v_a], dim=1)

            mixed_a = self._mixed_attention(
                q_cat=q_a,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )

            x_action = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=res_a,
                gate_msa=gate_a,
                shift_mlp=sh_a,
                scale_mlp=sc_a,
                gate_mlp=g_a,
                use_gradient_checkpointing=gc_a,
                mixed_slice=mixed_a,
                context_payload=action_context,
            )

        return {
            VIDEO_EXPERT_KEY: x_video,
            ACTION_EXPERT_KEY: x_action,
        }

    def prefill_video_kv(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the video expert and return one K/V cache per action layer.

        Each entry contains the selected video source with tensors [B, Sv, H * Dh]."""
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"video_attention_mask must be 2D [Sv, Sv], got shape "
                f"{tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"video_attention_mask must be square, got shape "
                f"{tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                f"video_attention_mask seq length {video_attention_mask.shape[0]} "
                f"!= video_tokens seq length {video_tokens.shape[1]}"
            )

        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        x = video_tokens


        # Non-fusion: run all video layers, capture KV from needed source layers.
        # fused_kv must capture EVERY layer: _select_or_mix_stacked_video_kv
        # index_selects with ORIGINAL layer indices (kv_source_indices) and
        # _stack_video_kv requires the full N-layer stack, exactly like the
        # training path (forward_decoupled). Capturing only the mapped layers
        # produced a compacted stack that raised (or mis-indexed) at inference
        # for any non-identity mapping while training worked. Selected modes
        # keep capturing only the mapped layers (they are consumed by original
        # index directly from captured_kv).
        if self.kv_source_mode in {"new_fused_kv"}:
            needed = set(range(self.video_num_layers))
        else:
            needed = set(self.kv_source_mapping)
        captured_kv: dict[int, dict[str, torch.Tensor]] = {}

        for layer_idx in range(self.video_num_layers):
            block = video_expert.blocks[layer_idx]
            (
                q, k, v,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v,
                attention_mask=video_attention_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )

            if layer_idx in needed:
                captured_kv[layer_idx] = {"k": k, "v": v}

        if self.kv_source_mode in {"new_fused_kv"}:
            all_kv = [captured_kv[i] for i in sorted(captured_kv)]
            stacked_k, stacked_v = self._stack_video_kv(all_kv)
            result = []
            for action_idx in range(self.action_num_layers):
                k_mixed, v_mixed = self._select_or_mix_stacked_video_kv(
                    stacked_k, stacked_v, action_idx,
                )
                result.append({"k": k_mixed, "v": v_mixed})
            return result

        return [captured_kv[src] for src in self.kv_source_mapping]

    def prepare_action_video_kv(
        self,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
    ) -> PreparedActionVideoKV:
        """Mark the already action-layer-indexed base cache as reusable.

        ``MoTDecoupled.prefill_video_kv`` already returns one entry per action
        layer.  Faster-WAM's RoPE-aware subclass overrides this method because
        its prefill cache retains all raw video layers until fusion.
        """
        if isinstance(video_kv_per_layer, PreparedActionVideoKV):
            return video_kv_per_layer
        if len(video_kv_per_layer) != self.action_num_layers:
            raise ValueError(
                f"video_kv_per_layer length {len(video_kv_per_layer)} != "
                f"action_num_layers {self.action_num_layers}"
            )
        return PreparedActionVideoKV(video_kv_per_layer)

    def prefill_video_final_kv(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Backward-compat wrapper. Prefer ``prefill_video_kv()`` instead."""
        result_list = self.prefill_video_kv(
            video_tokens, video_freqs, video_t_mod,
            video_context_payload, video_attention_mask,
        )
        return result_list[-1]

    def _stack_video_kv(
        self,
        video_kv_cache: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and stack flattened video KV as [layer, B, S, H, Dh]."""
        if len(video_kv_cache) != self.video_num_layers:
            raise ValueError(
                f"video_kv_cache length {len(video_kv_cache)} != "
                f"video_num_layers {self.video_num_layers}"
            )

        expected_hidden = self.num_heads * self.attn_head_dim
        k_per_layer: list[torch.Tensor] = []
        v_per_layer: list[torch.Tensor] = []
        expected_shape: torch.Size | None = None
        for layer_idx, layer_kv in enumerate(video_kv_cache):
            if "k" not in layer_kv or "v" not in layer_kv:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] must contain 'k' and 'v'."
                )
            k = layer_kv["k"]
            v = layer_kv["v"]
            if k.shape != v.shape:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] K/V shape mismatch: "
                    f"{tuple(k.shape)} vs {tuple(v.shape)}"
                )
            if k.ndim != 3 or k.shape[-1] != expected_hidden:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] tensors must be [B, Sv, "
                    f"{expected_hidden}], got {tuple(k.shape)}"
                )
            if expected_shape is None:
                expected_shape = k.shape
            elif k.shape != expected_shape:
                raise ValueError(
                    f"video_kv_cache[{layer_idx}] shape {tuple(k.shape)} != "
                    f"first layer shape {tuple(expected_shape)}"
                )
            k_per_layer.append(k.unflatten(-1, (self.num_heads, self.attn_head_dim)))
            v_per_layer.append(v.unflatten(-1, (self.num_heads, self.attn_head_dim)))

        return torch.stack(k_per_layer), torch.stack(v_per_layer)

    def _select_or_mix_stacked_video_kv(
        self,
        stacked_k: torch.Tensor,
        stacked_v: torch.Tensor,
        action_layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select final-layer K/V or mix all sources with one distribution per head."""
        if self.kv_source_mode != "new_fused_kv":
            source_idx = self.kv_source_mapping[action_layer_idx]
            return stacked_k[source_idx].flatten(-2), stacked_v[source_idx].flatten(-2)
        source_indices = self.kv_source_indices.to(device=stacked_k.device)
        selected_k = stacked_k.index_select(0, source_indices)
        selected_v = stacked_v.index_select(0, source_indices)
        weights = self.lite_kv_fusion_logits[:, :, action_layer_idx].float().softmax(dim=1)
        weights = weights.to(device=selected_k.device, dtype=selected_k.dtype)
        mixed_k = torch.einsum("lbshd,hl->bshd", selected_k, weights).flatten(-2)
        mixed_v = torch.einsum("lbshd,hl->bshd", selected_v, weights).flatten(-2)
        if self.k_fused_norm is not None:
            mixed_k = self.k_fused_norm[action_layer_idx](mixed_k)
        return mixed_k, mixed_v

    def forward_action_with_video_kv(
        self,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run action expert using per-action-layer K/V from prefill_video_kv.

        Args:
            video_kv_per_layer: List of length ``action_num_layers``, each a
                dict with ``"k"`` and ``"v"`` tensors of shape
                ``[B, Sv, H*Dh]``. Source selection or fusion is already done
                by ``prefill_video_kv()``.
            action_tokens: Action tokens before layer 0, shape ``[B, Sa, D]``.
            action_freqs: Action RoPE frequencies.
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional cross-attention context dict.
            action_attention_mask: Rectangular mask ``[Sa, Sv + Sa]``.

        Returns:
            Updated action tokens, shape ``[B, Sa, D_action]``.
        """
        if len(video_kv_per_layer) != self.action_num_layers:
            raise ValueError(
                f"video_kv_per_layer length {len(video_kv_per_layer)} != "
                f"action_num_layers {self.action_num_layers}"
            )
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        x = action_tokens

        for layer_idx in range(self.action_num_layers):
            kv = video_kv_per_layer[layer_idx]
            k_video = kv["k"]
            v_video = kv["v"]

            if layer_idx == 0:
                video_seq_len = k_video.shape[1]
                action_seq_len = action_tokens.shape[1]
                expected_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
                if action_attention_mask.ndim != 2:
                    raise ValueError(
                        f"action_attention_mask must be 2D, got ndim={action_attention_mask.ndim}"
                    )
                if action_attention_mask.shape != torch.Size(expected_mask_shape):
                    raise ValueError(
                        f"action_attention_mask shape must be {expected_mask_shape}, "
                        f"got {tuple(action_attention_mask.shape)}"
                    )

            block = action_expert.blocks[layer_idx]
            (
                q_action, k_action, v_action,
                residual_x,
                gate_msa,
                shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)

            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )

        return x

    def forward_action_with_final_kv(
        self,
        final_video_kv,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Backward-compat wrapper. Prefer ``forward_action_with_video_kv()`` instead."""
        if isinstance(final_video_kv, dict):
            if "k" not in final_video_kv or "v" not in final_video_kv:
                raise ValueError(
                    "final_video_kv must contain 'k' and 'v' keys."
                )
            video_kv_per_layer = [final_video_kv] * self.action_num_layers
        else:
            video_kv_per_layer = final_video_kv
        return self.forward_action_with_video_kv(
            video_kv_per_layer=video_kv_per_layer,
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload=action_context_payload,
            action_attention_mask=action_attention_mask,
        )
