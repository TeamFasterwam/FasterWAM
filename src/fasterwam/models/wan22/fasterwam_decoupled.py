"""RoPE-aware video-layer fusion for the published action readouts."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from fasterwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .fastwam_decoupled import FastWAMDecoupled
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot_decoupled import (
    ACTION_EXPERT_KEY,
    VIDEO_EXPERT_KEY,
    MoTDecoupled,
    PreparedActionVideoKV,
)
from .wan_video_dit import modulate, precompute_freqs_cis, rope_apply

logger = get_logger(__name__)


def _validate_new_fused_kv_projection_mode(
    kv_source_mode: str,
    projection_mode: str,
) -> str:
    """Validate and normalize the projection-mode configuration."""
    projection_mode = str(projection_mode)
    if projection_mode not in MoTDecoupled.supported_new_fused_kv_projection_modes:
        raise ValueError(
            f"Unknown new_fused_kv_projection_mode {projection_mode!r}; "
            "expected one of "
            f"{sorted(MoTDecoupled.supported_new_fused_kv_projection_modes)}"
        )
    if kv_source_mode != "new_fused_kv" and projection_mode != "full":
        raise ValueError(
            "new_fused_kv_projection_mode is only configurable when "
            "kv_source_mode=new_fused_kv; use full for other modes."
        )
    return projection_mode


class MoTDecoupledActionAlignedVideoRoPE(MoTDecoupled):
    """Decoupled MoT with raw video K cache and action-aligned video K RoPE.

    Differences from ``MoTDecoupled``:
    - video self-attention uses native video 3D RoPE exactly as before;
    - cached video K for action attention is stored before 3D RoPE by default
      (``legacy_3d`` instead caches post-3D-RoPE K);
    - before concatenating ``[video K, action K]``, video K is re-positioned
      according to the selected KV/RoPE mode.
    """

    enable_new_fused_kv_key_norm = True
    supported_new_fused_kv_rope_modes = {
        "aligned_3d",
        "original_3d",
        "legacy_3d",
    }
    # Normalized (h, w) camera centers for the supported image mosaics.
    aligned_3d_action_spatial_anchor_presets = {
        "center": ((0.5, 0.5),),
        "horizontal": ((0.5, 0.25), (0.5, 0.75)),
        "libero": ((0.5, 0.25), (0.5, 0.75)),
        "robotwin": (
            (1.0 / 3.0, 0.5),
            (5.0 / 6.0, 0.25),
            (5.0 / 6.0, 0.75),
        ),
    }
    _aligned_3d_family_rope_modes = frozenset({"aligned_3d"})

    def __init__(
        self,
        *args,
        new_fused_kv_rope_mode: str = "aligned_3d",
        aligned_3d_action_spatial_anchor_layout: str | None = "center",
        **kwargs,
    ):
        if new_fused_kv_rope_mode not in self.supported_new_fused_kv_rope_modes:
            raise ValueError(
                "Unknown new_fused_kv_rope_mode "
                f"{new_fused_kv_rope_mode!r}; expected one of "
                f"{sorted(self.supported_new_fused_kv_rope_modes)}"
            )
        self.new_fused_kv_rope_mode = new_fused_kv_rope_mode
        anchor_layout = (
            "center"
            if aligned_3d_action_spatial_anchor_layout is None
            else str(aligned_3d_action_spatial_anchor_layout)
        )
        if anchor_layout not in self.aligned_3d_action_spatial_anchor_presets:
            raise ValueError(
                "Unknown aligned_3d_action_spatial_anchor_layout "
                f"{anchor_layout!r}; expected one of "
                f"{sorted(self.aligned_3d_action_spatial_anchor_presets)}"
            )
        self.aligned_3d_action_spatial_anchor_layout = anchor_layout
        self.aligned_3d_action_spatial_anchors = (
            self.aligned_3d_action_spatial_anchor_presets[anchor_layout]
        )
        super().__init__(*args, **kwargs)

    @property
    def _cache_post_3d_rope_video_k(self) -> bool:
        """Whether action-path video K should be cached after video 3D RoPE."""
        return (
            self.kv_source_mode == "new_fused_kv"
            and self.new_fused_kv_rope_mode == "legacy_3d"
        )

    def _build_expert_attention_io_with_raw_k(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build attention tensors and expose both raw and RoPE-applied K."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k_raw = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k_rope = rope_apply(k_raw, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k_rope,
            k_raw,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _build_expert_attention_io_with_pre_norm_k(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build video attention while exposing K before RMSNorm and RoPE."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k_pre_norm = block.self_attn.k(attn_input)
        k_rope = rope_apply(block.self_attn.norm_k(k_pre_norm), freqs, block.num_heads)
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k_rope,
            k_pre_norm,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_action_zero_rope_to_video_k(
        self,
        k_video_raw: torch.Tensor,
        action_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate every video key token with action token 0's 1D RoPE."""
        if action_freqs.ndim != 3 or action_freqs.shape[0] < 1:
            raise ValueError(
                f"action_freqs must be [Sa, 1, rope_dim] with Sa >= 1, "
                f"got {tuple(action_freqs.shape)}"
            )
        action_zero_freq = action_freqs[:1].to(device=k_video_raw.device)
        return rope_apply(k_video_raw, action_zero_freq, self.num_heads)

    def _split_3d_rope_dims(self) -> tuple[int, int, int]:
        """Return temporal/height/width RoPE complex dims for one attention head."""
        temporal_real_dim = self.attn_head_dim - 2 * (self.attn_head_dim // 3)
        spatial_real_dim = self.attn_head_dim // 3
        return (
            temporal_real_dim // 2,
            spatial_real_dim // 2,
            spatial_real_dim // 2,
        )

    def _infer_video_tokens_per_frame(self, video_freqs: torch.Tensor) -> int:
        """Infer the contiguous video tokens-per-frame from repeated temporal freqs."""
        t_dim, _, _ = self._split_3d_rope_dims()
        if video_freqs.ndim != 3 or video_freqs.shape[0] < 1:
            raise ValueError(
                f"video_freqs must be [Sv, 1, rope_dim], got {tuple(video_freqs.shape)}"
            )
        temporal = video_freqs[:, 0, :t_dim]
        first_temporal = temporal[:1]
        same_as_first = torch.isclose(
            temporal,
            first_temporal.expand_as(temporal),
        ).all(dim=-1)
        first_different = (~same_as_first).nonzero(as_tuple=False)
        if first_different.numel() == 0:
            return int(video_freqs.shape[0])
        return int(first_different[0, 0].item())

    def _infer_video_spatial_grid_size(
        self,
        video_freqs: torch.Tensor,
        tokens_per_frame: int,
    ) -> tuple[int, int]:
        """Infer one-frame token grid height/width from the 3D RoPE layout."""
        if tokens_per_frame <= 1:
            return 1, max(tokens_per_frame, 1)
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        first_frame = video_freqs[:tokens_per_frame, 0]
        w_start = t_dim + h_dim
        w_part = first_frame[:, w_start:w_start + w_dim]
        width = None
        for idx in range(1, tokens_per_frame + 1):
            if tokens_per_frame % idx != 0:
                continue
            if torch.allclose(w_part[idx - 1], w_part[-1]):
                width = idx
                break
        if width is None or width <= 0:
            width = tokens_per_frame
        return max(tokens_per_frame // width, 1), width


    def _allocate_aligned_3d_head_anchor_indices(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate action-attention heads to camera-view anchors.

        Single-anchor layouts map every head to anchor 0. Multi-anchor layouts
        give half the heads to the main camera (anchor 0) and divide the rest
        evenly among wrist cameras (anchors 1..C-1).
        """
        num_anchors = len(self.aligned_3d_action_spatial_anchors)
        if num_anchors == 1:
            return torch.zeros(
                self.num_heads, device=device, dtype=torch.long
            )
        if self.num_heads % 2 != 0:
            raise ValueError(
                "aligned_3d main-half head allocation requires an even "
                f"num_heads, got {self.num_heads}."
            )
        num_main_heads = self.num_heads // 2
        num_wrist_heads = self.num_heads - num_main_heads
        num_wrists = num_anchors - 1
        wrist_base, wrist_remainder = divmod(num_wrist_heads, num_wrists)
        head_anchor_indices_list = [0] * num_main_heads
        for wrist_idx in range(num_wrists):
            wrist_count = wrist_base + int(wrist_idx < wrist_remainder)
            head_anchor_indices_list.extend([wrist_idx + 1] * wrist_count)
        return torch.tensor(
            head_anchor_indices_list,
            device=device,
            dtype=torch.long,
        )


    def _build_new_fused_kv_3d_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build fixed-camera spatial phases with unscaled action time.

        Video time is anchored at zero. The DoT subclass replaces the per-head camera
        assignment with cyclic frequency interleaving for the lifted 3D preset."""
        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        expected_rope_dim = t_dim + h_dim + w_dim
        if video_freqs.ndim != 3 or video_freqs.shape[1] != 1:
            raise ValueError(
                f"video_freqs must be [Sv, 1, rope_dim], got {tuple(video_freqs.shape)}"
            )
        if video_freqs.shape[-1] != expected_rope_dim:
            raise ValueError(
                f"video_freqs rope dim {video_freqs.shape[-1]} != expected "
                f"{expected_rope_dim} for attn_head_dim={self.attn_head_dim}"
            )

        temporal_freqs = precompute_freqs_cis(
            self.attn_head_dim - 2 * (self.attn_head_dim // 3),
            end=max(action_seq_len, 1),
        ).to(device=video_freqs.device)
        video_rope_freqs = video_freqs.clone()
        # Published lifted RoPE uses unscaled action positions and video time zero.
        video_anchor_temporal = temporal_freqs[:1].to(dtype=video_freqs.dtype)
        video_rope_freqs[:, :, :t_dim] = video_anchor_temporal.view(1, 1, t_dim)

        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs=video_freqs,
            tokens_per_frame=tokens_per_frame,
        )
        if height * width != tokens_per_frame:
            raise ValueError(
                f"Invalid inferred video spatial grid {(height, width)} for "
                f"tokens_per_frame={tokens_per_frame}."
            )

        # Recover the supplied video's spatial frequency bases at coordinate 1.
        # Their angles are all in [0, 1], so torch.angle does not wrap them.
        if height > 1:
            h_unit = video_freqs[width, 0, t_dim:t_dim + h_dim]
        else:
            h_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=video_freqs.device, dtype=video_freqs.dtype)
        if width > 1:
            w_unit = video_freqs[1, 0, t_dim + h_dim:]
        else:
            w_unit = precompute_freqs_cis(
                self.attn_head_dim // 3, end=2
            )[1].to(device=video_freqs.device, dtype=video_freqs.dtype)

        anchor_positions = torch.tensor(
            self.aligned_3d_action_spatial_anchors,
            device=video_freqs.device,
            dtype=torch.angle(h_unit).dtype,
        )
        anchor_h_coordinates = anchor_positions[:, 0] * height - 0.5
        anchor_w_coordinates = anchor_positions[:, 1] * width - 0.5
        h_angles = anchor_h_coordinates[:, None] * torch.angle(h_unit)[None, :]
        w_angles = anchor_w_coordinates[:, None] * torch.angle(w_unit)[None, :]
        anchor_h_freqs = torch.polar(torch.ones_like(h_angles), h_angles)
        anchor_w_freqs = torch.polar(torch.ones_like(w_angles), w_angles)
        anchor_spatial = torch.cat([anchor_h_freqs, anchor_w_freqs], dim=-1)
        head_anchor_indices = self._allocate_aligned_3d_head_anchor_indices(
            device=video_freqs.device,
        )
        action_spatial = anchor_spatial.index_select(0, head_anchor_indices)

        action_temporal = temporal_freqs[:action_seq_len].to(dtype=video_freqs.dtype)
        action_rope_freqs = torch.cat(
            [
                action_temporal.view(action_seq_len, 1, t_dim).expand(
                    -1, self.num_heads, -1
                ),
                action_spatial.unsqueeze(0).expand(action_seq_len, -1, -1),
            ],
            dim=-1,
        )
        return video_rope_freqs, action_rope_freqs

    # Training and inference share the video loop and action attention.
    def _validate_decoupled_forward_inputs(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate per-expert forward inputs; return (video_mask, action_mask)."""
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
        expected_action_mask_shape = (action_seq_len, video_seq_len + action_seq_len)
        if action_mask.shape != torch.Size(expected_action_mask_shape):
            raise ValueError(
                f"Action attention mask must have shape {expected_action_mask_shape}, "
                f"got {tuple(action_mask.shape)}"
            )
        return video_mask, action_mask

    def _run_video_layers_raw_kv(
        self,
        x: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context: Optional[dict],
        video_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Run every video layer (3D-RoPE'd self-attn), caching video K and V.

        By default the cached K is pre-RoPE (raw / pre-norm, depending on the
        projection mode). ``legacy_3d`` instead caches the post-3D-RoPE K used
        in video self-attention, so fusion sees already-positioned keys.

        The full post-block MUST run each layer because layer i+1's K depends
        on layer i's post-block output.

        Returns:
            (final video tokens, per-layer K list, per-layer V list);
            cached tensors are flattened ``[B, Sv, H*Dh]``.
        """
        video_expert = self.mixtures[VIDEO_EXPERT_KEY]
        raw_k_per_layer: list[torch.Tensor] = []
        raw_v_per_layer: list[torch.Tensor] = []
        cache_post_rope_k = self._cache_post_3d_rope_video_k
        for video_layer_idx in range(self.video_num_layers):
            video_block = video_expert.blocks[video_layer_idx]
            build_attention_io = (
                self._build_expert_attention_io_with_raw_k
                if self.kv_source_mode != "new_fused_kv"
                else self._build_expert_attention_io_with_pre_norm_k
            )
            (
                q_video,
                k_video_rope,
                k_video_raw,
                v_video,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = build_attention_io(
                expert=video_expert,
                block=video_block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            mixed = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_video_rope,
                v_cat=v_video,
                attention_mask=video_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
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

            raw_k_per_layer.append(
                k_video_rope if cache_post_rope_k else k_video_raw
            )
            raw_v_per_layer.append(v_video)

        return x, raw_k_per_layer, raw_v_per_layer

    def _rotate_action_video_key(
        self,
        key: torch.Tensor,
        frequencies: torch.Tensor | None,
        *,
        video_freqs: torch.Tensor | None,
    ) -> torch.Tensor:
        return key if frequencies is None else rope_apply(key, frequencies, self.num_heads)

    def _resolve_action_interface_rope(
        self,
        video_rope_freqs: torch.Tensor | None,
        action_rope_freqs: torch.Tensor,
        *,
        video_freqs: torch.Tensor | None,
        action_freqs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Resolve the positional interface after selecting the shared routing."""
        return video_rope_freqs, action_rope_freqs

    def _run_action_layers_with_video_kv_source(
        self,
        x: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: Optional[dict],
        action_mask: torch.Tensor,
        video_kv_for_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
        validate_mask_at_layer0: bool = False,
        video_freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run action layers against selected or fused video K/V.

        Original 3D phases apply after fusion; native and retracted interfaces consume
        keys cached after video RoPE. Lifted action phases reuse the video spatial basis.
        The DoT interface hook selects the published positional convention."""
        action_expert = self.mixtures[ACTION_EXPERT_KEY]
        use_new_fused_kv = self.kv_source_mode == "new_fused_kv"
        use_aligned_new_fused_kv_rope = (
            use_new_fused_kv
            and self.new_fused_kv_rope_mode in self._aligned_3d_family_rope_modes
        )
        use_original_new_fused_kv_rope = (
            use_new_fused_kv and self.new_fused_kv_rope_mode == "original_3d"
        )
        # legacy_3d caches native post-RoPE keys; the interface may rephase them.
        use_skip_action_path_video_rope = (
            use_new_fused_kv
            and self.new_fused_kv_rope_mode == "legacy_3d"
        )
        if use_aligned_new_fused_kv_rope:
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv fixed-RoPE action path requires video_freqs "
                    "to build video-style 3D RoPE for action attention."
                )
            video_rope_freqs, action_rope_freqs = self._build_new_fused_kv_3d_rope_freqs(
                video_freqs=video_freqs,
                action_seq_len=x.shape[1],
            )
        elif use_original_new_fused_kv_rope:
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv original-3D-RoPE action path requires video_freqs "
                    "to apply the original video 3D RoPE to fused video K."
                )
            # Rank/rope-dim validation for the original_3d branch. Unlike the
            # aligned_3d path, original_3d uses video_freqs verbatim as the RoPE
            # basis for rope_apply(k_video_raw, video_rope_freqs, num_heads), so a
            # malformed freqs tensor (wrong rank, or a last dim that does not match
            # attn_head_dim // 2) would silently mis-rotate every video token. This
            # mirrors the shape contract asserted by the aligned-path builder
            # _build_new_fused_kv_3d_rope_freqs (t_dim + h_dim + w_dim). It is
            # complementary to the 5fb5881 length guard (freqs.shape[0] vs Sv),
            # which is enforced separately at the per-layer rope_apply site.
            t_dim, h_dim, w_dim = self._split_3d_rope_dims()
            expected_rope_dim = t_dim + h_dim + w_dim
            if video_freqs.ndim != 3 or video_freqs.shape[1] != 1:
                raise ValueError(
                    "new_fused_kv original_3d video_freqs must be "
                    "[Sv, 1, rope_dim], got "
                    f"{tuple(video_freqs.shape)}"
                )
            if video_freqs.shape[-1] != expected_rope_dim:
                raise ValueError(
                    "new_fused_kv original_3d video_freqs rope dim "
                    f"{video_freqs.shape[-1]} != expected {expected_rope_dim} "
                    f"for attn_head_dim={self.attn_head_dim}"
                )
            video_rope_freqs = video_freqs
            action_rope_freqs = action_freqs
        elif use_skip_action_path_video_rope:
            video_rope_freqs = None
            action_rope_freqs = action_freqs
        else:
            video_rope_freqs = action_freqs[:1]
            action_rope_freqs = action_freqs

        video_rope_freqs, action_rope_freqs = self._resolve_action_interface_rope(
            video_rope_freqs, action_rope_freqs,
            video_freqs=video_freqs, action_freqs=action_freqs,
        )

        for action_layer_idx in range(self.action_num_layers):
            k_video_raw, v_video = video_kv_for_layer(action_layer_idx)
            if use_new_fused_kv:
                k_video = self._rotate_action_video_key(
                    k_video_raw, video_rope_freqs, video_freqs=video_freqs,
                )
            else:
                k_video = self._apply_action_zero_rope_to_video_k(
                    k_video_raw=k_video_raw,
                    action_freqs=video_rope_freqs,
                )

            if validate_mask_at_layer0 and action_layer_idx == 0:
                video_seq_len = k_video.shape[1]
                action_seq_len = x.shape[1]
                batch_size = int(x.shape[0])
                expected_2d = (action_seq_len, video_seq_len + action_seq_len)
                expected_3d = (
                    self.num_heads,
                    action_seq_len,
                    video_seq_len + action_seq_len,
                )
                # SDPA also accepts per-head and batched visibility masks.
                expected_4d = (
                    batch_size,
                    self.num_heads,
                    action_seq_len,
                    video_seq_len + action_seq_len,
                )
                if action_mask.shape not in (
                    torch.Size(expected_2d),
                    torch.Size(expected_3d),
                    torch.Size(expected_4d),
                ):
                    raise ValueError(
                        "action_attention_mask shape must be "
                        f"{expected_2d}, {expected_3d}, or {expected_4d}, "
                        f"got {tuple(action_mask.shape)}"
                    )

            action_block = action_expert.blocks[action_layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x,
                freqs=action_rope_freqs,
                t_mod=action_t_mod,
            )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)

            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_mask,
            )

            x = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context,
            )

        return x

    def forward_decoupled(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_masks: Dict[str, torch.Tensor],
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Training forward with raw video-K fusion and action-aligned RoPE."""
        video_mask, action_mask = self._validate_decoupled_forward_inputs(
            embeds_all=embeds_all,
            attention_masks=attention_masks,
            freqs_all=freqs_all,
            t_mod_all=t_mod_all,
        )

        x_video, raw_k_per_layer, raw_v_per_layer = self._run_video_layers_raw_kv(
            x=embeds_all[VIDEO_EXPERT_KEY],
            video_freqs=freqs_all[VIDEO_EXPERT_KEY],
            video_t_mod=t_mod_all[VIDEO_EXPERT_KEY],
            video_context=context_all.get(VIDEO_EXPERT_KEY),
            video_mask=video_mask,
        )

        video_kv_cache = [
            {"k": k, "v": v} for k, v in zip(raw_k_per_layer, raw_v_per_layer)
        ]
        stacked_k, stacked_v = self._stack_video_kv(video_kv_cache)

        x_action = self._run_action_layers_with_video_kv_source(
            x=embeds_all[ACTION_EXPERT_KEY],
            action_freqs=freqs_all[ACTION_EXPERT_KEY],
            action_t_mod=t_mod_all[ACTION_EXPERT_KEY],
            action_context=context_all.get(ACTION_EXPERT_KEY),
            action_mask=action_mask,
            video_kv_for_layer=lambda idx: self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=idx,
            ),
            video_freqs=freqs_all[VIDEO_EXPERT_KEY],
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
        """Run video layers and retain one K/V cache per video layer.

        Each entry has tensors [B, Sv, H * Dh] and the native video frequencies.
        Original-3D fusion uses pre-norm keys; post-RoPE fusion caches the normalized,
        rotated keys. Values are never rotated."""
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

        # Run the full video block per layer (self-attn uses 3D-RoPE'd K),
        # caching K/V of every layer via the shared training helper. Default
        # modes cache RAW K; ``legacy_3d`` caches post-3D-RoPE K.
        _, raw_k_per_layer, raw_v_per_layer = self._run_video_layers_raw_kv(
            x=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context_payload,
            video_mask=video_attention_mask,
        )


        return [
            {"k": k, "v": v, "freqs": video_freqs}
            if self.kv_source_mode == "new_fused_kv"
            else {"k": k, "v": v}
            for k, v in zip(raw_k_per_layer, raw_v_per_layer)
        ]

    def forward_action_with_video_kv(
        self,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise action tokens using raw per-video-layer or prepared per-action-layer KV.

        A prepared cache reuses static fusion across denoising steps. Raw caches are
        accepted for direct callers and are fused through the same attention helper."""

        if isinstance(video_kv_per_layer, PreparedActionVideoKV):
            if len(video_kv_per_layer) != self.action_num_layers:
                raise ValueError(
                    f"video_kv_per_layer length {len(video_kv_per_layer)} != "
                    f"action_num_layers {self.action_num_layers}"
                )
            prepared_video_kv = video_kv_per_layer
            video_freqs = None
            if self.kv_source_mode == "new_fused_kv":
                video_freqs = prepared_video_kv[0].get("freqs")
                if video_freqs is None:
                    raise ValueError(
                        "prepared new_fused_kv cache must include `freqs`; "
                        "use prepare_action_video_kv() on a cache from the "
                        "same model."
                    )
            return self._run_action_layers_with_video_kv_source(
                x=action_tokens,
                action_freqs=action_freqs,
                action_t_mod=action_t_mod,
                action_context=action_context_payload,
                action_mask=action_attention_mask,
                video_kv_for_layer=lambda idx: (
                    prepared_video_kv[idx]["k"],
                    prepared_video_kv[idx]["v"],
                ),
                validate_mask_at_layer0=True,
                video_freqs=video_freqs,
            )

        # Backward-compatible direct-call path. Production inference calls
        # prepare_action_video_kv() once before entering the denoising loop.
        stacked_k, stacked_v = self._stack_video_kv(video_kv_per_layer)
        video_freqs = None
        if self.kv_source_mode == "new_fused_kv":
            video_freqs = video_kv_per_layer[0].get("freqs")
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv fixed-RoPE inference cache must include "
                    "`freqs`; use prefill_video_kv() from the same model."
                )
            # Sequence-axis derivation: the cached sidecar is consumed by
            # rope_apply(k_video_raw, video_rope_freqs, num_heads) at
            # _run_action_layers_with_video_kv_source line ~414. There
            # k_video_raw is [B, Sv, D] -> rearranged to [B, Sv, n, d] and the
            # freqs (video_rope_freqs = video_freqs.clone(), so same axis-0
            # length) broadcast as [Sv, 1, rope_dim/2] against [B, Sv, n, d/2].
            # rope_apply broadcasts positionally, so a shorter/longer freqs
            # axis-0 would either error or (when equal by coincidence) silently
            # mis-rotate; axis 0 of video_freqs MUST equal the cached video K
            # sequence length Sv. _build_new_fused_kv_3d_rope_freqs asserts
            # video_freqs is [Sv, 1, rope_dim] (line ~217), fixing Sv on axis 0.
            # stacked_k is [N, B, Sv, num_heads, attn_head_dim] from
            # _stack_video_kv, so Sv = stacked_k.shape[2].
            cached_video_seq_len = stacked_k.shape[2]
            freqs_seq_len = video_freqs.shape[0]
            if freqs_seq_len != cached_video_seq_len:
                raise ValueError(
                    "new_fused_kv cached `freqs` sequence length "
                    f"{freqs_seq_len} != cached video K sequence length "
                    f"{cached_video_seq_len}; the freqs sidecar is malformed or "
                    "stale. Regenerate the cache with prefill_video_kv() from "
                    "the same model."
                )
        return self._run_action_layers_with_video_kv_source(
            x=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context_payload,
            action_mask=action_attention_mask,
            video_kv_for_layer=lambda idx: self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=idx,
            ),
            validate_mask_at_layer0=True,
            video_freqs=video_freqs,
        )

    def prepare_action_video_kv(
        self,
        video_kv_per_layer: list[dict[str, torch.Tensor]],
    ) -> PreparedActionVideoKV:
        """Select or fuse static video K/V once for all action denoising steps.

        The native frequency sidecar is retained for the positional interface."""
        if isinstance(video_kv_per_layer, PreparedActionVideoKV):
            return video_kv_per_layer

        stacked_k, stacked_v = self._stack_video_kv(video_kv_per_layer)
        video_freqs = None
        if self.kv_source_mode == "new_fused_kv":
            video_freqs = video_kv_per_layer[0].get("freqs")
            if video_freqs is None:
                raise ValueError(
                    "new_fused_kv cache must include `freqs`; use "
                    "prefill_video_kv() from the same model."
                )
            if video_freqs.shape[0] != stacked_k.shape[2]:
                raise ValueError(
                    "new_fused_kv cached `freqs` sequence length "
                    f"{video_freqs.shape[0]} != cached video K sequence length "
                    f"{stacked_k.shape[2]}; regenerate the cache with "
                    "prefill_video_kv() from the same model."
                )

        prepared = PreparedActionVideoKV()
        for action_layer_idx in range(self.action_num_layers):
            k_video, v_video = self._select_or_mix_stacked_video_kv(
                stacked_k=stacked_k,
                stacked_v=stacked_v,
                action_layer_idx=action_layer_idx,
            )
            entry = {"k": k_video, "v": v_video}
            if video_freqs is not None:
                entry["freqs"] = video_freqs
            prepared.append(entry)
        return prepared


class FasterWAMDecoupled(FastWAMDecoupled):
    """FastWAMDecoupled variant using action-aligned video K RoPE."""

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        decoupled: bool = True,
        kv_source_mapping: list[int] | None = None,
        kv_source_mode: str = "final_only",
        fixed_rope: bool = True,
        new_fused_kv_rope_mode: str = "aligned_3d",
        aligned_3d_action_spatial_anchor_layout: str | None = "center",
        new_fused_kv_projection_mode: str = "full",
        new_fused_kv_simple_head_softmax_fuse_mode: str = "all",
        _mot_class: type[MoTDecoupledActionAlignedVideoRoPE] | None = None,
        _mot_options: dict[str, Any] | None = None,
    ):
        if video_dit_config is None:
            raise ValueError(
                "`video_dit_config` is required for "
                "FasterWAMDecoupled.from_wan22_pretrained()."
            )
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")
        if not decoupled:
            raise ValueError("FasterWAMDecoupled requires decoupled=True.")
        new_fused_kv_projection_mode = _validate_new_fused_kv_projection_mode(
            kv_source_mode,
            new_fused_kv_projection_mode,
        )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        layer_init_mapping = kv_source_mapping
        # All-layer fusion needs a separate source mapping for pretrained
        # action-backbone initialization.
        if kv_source_mode in ("new_fused_kv",):
            from .mot_decoupled import compute_kv_source_mapping

            action_config_for_init = action_dit_config or {}
            action_num_layers = int(action_config_for_init.get("num_layers", 5))
            if kv_source_mapping is not None and len(kv_source_mapping) == action_num_layers:
                layer_init_mapping = kv_source_mapping
            else:
                layer_init_mapping = compute_kv_source_mapping(
                    mode="uniform_end",
                    video_num_layers=len(video_expert.blocks),
                    action_num_layers=action_num_layers,
                )

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
            layer_init_mapping=layer_init_mapping,
        )

        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for mixed attention."
            )

        mot_cls = (
            MoTDecoupledActionAlignedVideoRoPE
            if fixed_rope or kv_source_mode == "new_fused_kv"
            else MoTDecoupled
        )
        if _mot_class is not None:
            if not issubclass(_mot_class, MoTDecoupledActionAlignedVideoRoPE):
                raise TypeError("_mot_class must implement the aligned-video MoT interface")
            mot_cls = _mot_class
        logger.info(
            "FasterWAMDecoupled fixed_rope=%s, kv_source_mode=%s, "
            "new_fused_kv_rope_mode=%s, aligned_3d_anchor_layout=%s, "
            "new_fused_kv_projection_mode=%s, "
            "simple_head_softmax_fuse_mode=%s, "
            "using MoT class %s",
            fixed_rope,
            kv_source_mode,
            new_fused_kv_rope_mode,
            aligned_3d_action_spatial_anchor_layout,
            new_fused_kv_projection_mode,
            new_fused_kv_simple_head_softmax_fuse_mode,
            mot_cls.__name__,
        )
        mot_kwargs = {}
        if issubclass(mot_cls, MoTDecoupledActionAlignedVideoRoPE):
            mot_kwargs["new_fused_kv_rope_mode"] = new_fused_kv_rope_mode
            mot_kwargs["aligned_3d_action_spatial_anchor_layout"] = (
                aligned_3d_action_spatial_anchor_layout
            )
        mot_kwargs["new_fused_kv_projection_mode"] = new_fused_kv_projection_mode
        if kv_source_mode == "new_fused_kv":
            mot_kwargs["new_fused_kv_simple_head_softmax_fuse_mode"] = (
                new_fused_kv_simple_head_softmax_fuse_mode
            )
        mot_kwargs.update(_mot_options or {})
        mot = mot_cls(
            mixtures={"video": video_expert, "action": action_expert},
            video_num_layers=len(video_expert.blocks),
            action_num_layers=len(action_expert.blocks),
            num_heads=int(video_expert.num_heads),
            attn_head_dim=int(video_expert.attn_head_dim),
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            kv_source_mapping=kv_source_mapping,
            kv_source_mode=kv_source_mode,
            **mot_kwargs,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model


def create_fasterwam_decoupled(
    model_id: str,
    tokenizer_model_id: str,
    tokenizer_max_len: int,
    load_text_encoder: bool,
    proprio_dim: int | None,
    redirect_common_files: bool,
    mot_checkpoint_mixed_attn: bool,
    action_dit_pretrained_path: str | None,
    skip_dit_load_from_pretrain: bool,
    decoupled: bool,
    kv_source_mode: str,
    video_dit_config: dict,
    action_dit_config: dict,
    video_scheduler: dict,
    action_scheduler: dict,
    loss: dict | None = None,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.bfloat16,
    fixed_rope: bool = True,
    new_fused_kv_rope_mode: str = "aligned_3d",
    aligned_3d_action_spatial_anchor_layout: str | None = "center",
    new_fused_kv_projection_mode: str = "full",
    new_fused_kv_simple_head_softmax_fuse_mode: str = "all",
    _mot_class: type[MoTDecoupledActionAlignedVideoRoPE] | None = None,
    _mot_options: dict[str, Any] | None = None,
) -> FasterWAMDecoupled:
    """Hydra factory for ``FasterWAMDecoupled``."""
    from omegaconf import DictConfig, OmegaConf
    from .mot_decoupled import compute_kv_source_mapping

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)

    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to dict, got {type(video_dit_config)}")
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to dict, got {type(action_dit_config)}")
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must resolve to dict, got {type(video_scheduler)}")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must resolve to dict, got {type(action_scheduler)}")
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must resolve to dict, got {type(loss)}")
    if not decoupled:
        raise ValueError("create_fasterwam_decoupled requires decoupled=True.")
    new_fused_kv_projection_mode = _validate_new_fused_kv_projection_mode(
        kv_source_mode,
        new_fused_kv_projection_mode,
    )

    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}"
        )

    action_dit_config = dict(action_dit_config)
    layer_selected = action_dit_config.pop("layer_selected", None)
    if layer_selected is not None:
        action_dit_config["num_layers"] = len(layer_selected)
        kv_source_mapping = list(layer_selected)
        logger.info(
            "layer_selected=%s, overriding num_layers=%d, kv_source_mapping=%s",
            layer_selected,
            len(layer_selected),
            kv_source_mapping,
        )
    else:
        kv_source_mapping = compute_kv_source_mapping(
            mode=kv_source_mode,
            video_num_layers=int(video_dit_config.get("num_layers", 30)),
            action_num_layers=int(action_dit_config.get("num_layers", 5)),
        )
        logger.info("KV source mode: %s, mapping: %s", kv_source_mode, kv_source_mapping)

    model = FasterWAMDecoupled.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        decoupled=True,
        kv_source_mapping=kv_source_mapping,
        kv_source_mode=kv_source_mode,
        fixed_rope=bool(fixed_rope),
        new_fused_kv_rope_mode=str(new_fused_kv_rope_mode),
        aligned_3d_action_spatial_anchor_layout=(
            None
            if aligned_3d_action_spatial_anchor_layout is None
            else str(aligned_3d_action_spatial_anchor_layout)
        ),
        new_fused_kv_projection_mode=new_fused_kv_projection_mode,
        new_fused_kv_simple_head_softmax_fuse_mode=str(
            new_fused_kv_simple_head_softmax_fuse_mode
        ),
        _mot_class=_mot_class,
        _mot_options=_mot_options,
    )
    return model
