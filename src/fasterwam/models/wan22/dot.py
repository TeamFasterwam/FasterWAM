"""Dock of Transformers (DoT): Lite KV-Fusion and the action positional interfaces.

The default reproduces Faster-WAM: Lite KV-Fusion mixes the native video keys
after QK-Norm and 3D RoPE, and retracted 1D-RoPE replaces their video phases
with the action position-zero phase. The native 3D and Lifted 3D RoPE
interfaces share this action loop, layer mixer and training/inference cache.
"""
from __future__ import annotations

from typing import Any

import torch

from .fasterwam_decoupled import MoTDecoupledActionAlignedVideoRoPE


class DockOfTransformers(MoTDecoupledActionAlignedVideoRoPE):
    """Action expert that reads all video layers through Lite KV-Fusion."""

    def __init__(
        self,
        *args: Any,
        key_aggregation_order: str = "norm_rope_then_fuse",
        position_interface: str = "retracted_1d",
        **kwargs: Any,
    ) -> None:
        self.key_aggregation_order = key_aggregation_order
        self.position_interface = position_interface
        super().__init__(*args, **kwargs)
        if key_aggregation_order == "norm_rope_then_fuse":
            # Allocate in the historical order, then remove this unused norm.
            # Its constant initialization does not consume RNG state.
            self.k_fused_norm = None
        if position_interface == "lifted_3d" and len(self.aligned_3d_action_spatial_anchors) < 2:
            raise ValueError("lifted_3d requires a multi-camera layout")

    @property
    def _cache_post_3d_rope_video_k(self) -> bool:
        return self.key_aggregation_order == "norm_rope_then_fuse"

    def _resolve_action_interface_rope(
        self,
        video_rope_freqs: torch.Tensor | None,
        action_rope_freqs: torch.Tensor,
        *,
        video_freqs: torch.Tensor | None,
        action_freqs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.position_interface == "lifted_3d":
            # The cached keys already use native video 3D phases; only action
            # Q/K use the lifted spatial pseudo-anchors.
            return None, action_rope_freqs
        if self.position_interface != "retracted_1d":
            return video_rope_freqs, action_rope_freqs
        rotary_pairs = self.attn_head_dim // 2
        if video_freqs is None:
            raise ValueError("retracted_1d requires the native video_freqs cache sidecar")
        if (
            video_freqs.ndim != 3
            or tuple(video_freqs.shape[1:]) != (1, rotary_pairs)
            or not torch.is_complex(video_freqs)
        ):
            raise ValueError(f"retracted_1d video_freqs must be complex [Sv, 1, {rotary_pairs}]")
        if (
            action_freqs.ndim != 3
            or action_freqs.shape[0] < 1
            or tuple(action_freqs.shape[1:]) != (1, rotary_pairs)
            or not torch.is_complex(action_freqs)
        ):
            raise ValueError(f"retracted_1d action_freqs must be complex [Sa, 1, {rotary_pairs}]")
        # Preserve Faster-WAM's fuse-then-rephase order and single model-dtype cast.
        rephase = video_freqs.conj() * action_freqs[:1].to(video_freqs.device)
        return rephase, action_freqs

    def _rotate_action_video_key(
        self,
        key: torch.Tensor,
        frequencies: torch.Tensor | None,
        *,
        video_freqs: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.position_interface == "retracted_1d":
            if video_freqs is None or key.shape[1] != video_freqs.shape[0]:
                raise ValueError("retracted_1d fused video K length does not match native video_freqs")
        return super()._rotate_action_video_key(key, frequencies, video_freqs=video_freqs)

    def _build_frequency_anchor_indices(
        self,
        *,
        num_anchors: int,
        num_spatial_frequencies: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Assign each (head, spatial frequency) pair to one camera pseudo-anchor.

        Three views (RoboTwin) use every permutation of (main, left wrist, right
        wrist) equally often across heads, so each view takes a third of the
        heads at every frequency. Other view counts use cyclic shifts,
        ``(frequency + head) mod C``; for two views this equals the two
        permutations of S2.
        """
        frequency_indices = torch.arange(
            num_spatial_frequencies, device=device, dtype=torch.long,
        )
        if num_anchors == 3:
            if self.num_heads % 6:
                raise ValueError(
                    "Lifted 3D RoPE with three views requires num_heads divisible by 6, "
                    f"got {self.num_heads}"
                )
            # Rows (0, 1), (2, 5) and (3, 4) differ by a left/right wrist swap,
            # so swapping the wrist views is equivalent to permuting heads.
            permutations = torch.tensor(
                [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], [2, 0, 1], [2, 1, 0]],
                device=device, dtype=torch.long,
            )
            head_permutations = permutations[
                torch.arange(self.num_heads, device=device).remainder(6)
            ]
            return head_permutations[:, frequency_indices.remainder(3)]
        head_phases = torch.arange(self.num_heads, device=device, dtype=torch.long)
        return (frequency_indices.unsqueeze(0) + head_phases.unsqueeze(1)).remainder(num_anchors)

    def _build_new_fused_kv_3d_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve the proven aligned_3d temporal and video-K construction.
        # Only the action spatial phase block is replaced below.
        video_rope_freqs, action_rope_freqs = (
            super()._build_new_fused_kv_3d_rope_freqs(
                video_freqs=video_freqs,
                action_seq_len=action_seq_len,
            )
        )

        anchors = self.aligned_3d_action_spatial_anchors
        num_anchors = len(anchors)
        if num_anchors < 2:
            raise ValueError(
                "lifted_3d RoPE requires at least two "
                "2D anchors; choose a multi-camera anchor layout"
            )

        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        if h_dim != w_dim:
            raise ValueError(
                "lifted_3d camera anchors require paired height/width "
                f"frequency counts, got h_dim={h_dim}, w_dim={w_dim}"
            )

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

        # Recover one-coordinate phase increments from the native Wan video
        # frequencies.  The production grids have both dimensions > 1; the
        # base builder already provides the fallback for degenerate grids, but
        # those grids cannot express distinct 2D camera anchors meaningfully.
        if height <= 1 or width <= 1:
            raise ValueError(
                "lifted_3d camera anchors require height and width > 1, "
                f"got {(height, width)}"
            )
        h_unit = video_freqs[width, 0, t_dim : t_dim + h_dim]
        w_unit = video_freqs[1, 0, t_dim + h_dim :]

        angle_dtype = torch.angle(h_unit).dtype
        anchor_positions = torch.tensor(
            anchors,
            device=video_freqs.device,
            dtype=angle_dtype,
        )
        anchor_h = anchor_positions[:, 0] * height - 0.5
        anchor_w = anchor_positions[:, 1] * width - 0.5

        # The assignment is based on the shared H/W frequency rank, not on the
        # flattened spatial-pair index.  Consequently H[r] and W[r] always
        # describe one coherent 2D anchor while consecutive frequency bands
        # alternate anchors inside the same head.
        frequency_anchor_indices = self._build_frequency_anchor_indices(
            num_anchors=num_anchors,
            num_spatial_frequencies=h_dim,
            device=video_freqs.device,
        )
        expected_shape = (self.num_heads, h_dim)
        if tuple(frequency_anchor_indices.shape) != expected_shape:
            raise ValueError(
                "frequency-anchor schedule must have shape "
                f"{expected_shape}, got {tuple(frequency_anchor_indices.shape)}"
            )
        if frequency_anchor_indices.dtype != torch.long:
            raise ValueError(
                "frequency-anchor schedule must use torch.long indices, got "
                f"{frequency_anchor_indices.dtype}"
            )
        invalid_anchor_indices = (frequency_anchor_indices < 0) | (
            frequency_anchor_indices >= num_anchors
        )
        if bool(invalid_anchor_indices.any()):
            raise ValueError(
                "frequency-anchor schedule contains an out-of-range anchor "
                f"index for num_anchors={num_anchors}"
            )

        h_coordinates = anchor_h[frequency_anchor_indices]
        w_coordinates = anchor_w[frequency_anchor_indices]
        h_angles = h_coordinates * torch.angle(h_unit)
        w_angles = w_coordinates * torch.angle(w_unit)
        interleaved_spatial = torch.cat(
            [
                torch.polar(torch.ones_like(h_angles), h_angles),
                torch.polar(torch.ones_like(w_angles), w_angles),
            ],
            dim=-1,
        ).to(dtype=video_freqs.dtype)

        # Multi-anchor contributions are summed inside each head's QK dot
        # product before the attention softmax.
        action_rope_freqs[:, :, t_dim:] = interleaved_spatial.unsqueeze(0)
        return video_rope_freqs, action_rope_freqs
