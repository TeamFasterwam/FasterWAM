"""Model factory for Faster-WAM, the Fast-WAM baseline, and the reported ablations."""
from __future__ import annotations

from numbers import Integral
from typing import Any


# Each tuple is (readout, key aggregation order, positional interface, TextCA); the
# rows are the steps of the ablation study (Fast-WAM, Final-layer, F -> N+R,
# N+R -> F, No TextCA, Lifted 3D RoPE, Retracted 1D RoPE).
SUPPORTED_MODEL_OPTIONS = frozenset({
    ("interleaved", "norm_rope_then_fuse", "native_3d", True),
    ("final_layer", "norm_rope_then_fuse", "native_3d", True),
    ("lite_kv_fusion", "fuse_then_norm_rope", "native_3d", True),
    ("lite_kv_fusion", "norm_rope_then_fuse", "native_3d", True),
    ("lite_kv_fusion", "norm_rope_then_fuse", "native_3d", False),
    ("lite_kv_fusion", "norm_rope_then_fuse", "lifted_3d", False),
    ("lite_kv_fusion", "norm_rope_then_fuse", "retracted_1d", False),
})

_COMMON_ARGUMENTS = frozenset({
    "model_id", "tokenizer_model_id", "tokenizer_max_len", "load_text_encoder",
    "proprio_dim", "redirect_common_files", "mot_checkpoint_mixed_attn",
    "action_dit_pretrained_path", "skip_dit_load_from_pretrain",
    "video_dit_config", "action_dit_config", "video_scheduler",
    "action_scheduler", "loss", "device", "model_dtype", "freeze_video_backbone",
})


def _validate_dimensions(readout: str, video: dict, action: dict) -> None:
    """Validate explicit dimensions without replacing constructor defaults."""
    dimension_names = ("hidden_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim")
    for config_name, config in (("video_dit_config", video), ("action_dit_config", action)):
        for name in dimension_names:
            # Missing fields retain the underlying constructor's validation
            # and defaults; the public entrypoint does not synthesize sizes.
            if name not in config:
                continue
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{config_name}.{name} must be a positive integer, got {value!r}")
        if "attn_head_dim" in config and config["attn_head_dim"] % 2:
            raise ValueError(f"{config_name}.attn_head_dim must be even for RoPE")
    for name in ("num_heads", "attn_head_dim"):
        if name in video and name in action and video[name] != action[name]:
            raise ValueError(f"action_dit_config.{name} must match video_dit_config.{name}")
    if "num_layers" not in video or "num_layers" not in action:
        return
    if readout == "interleaved" and action["num_layers"] != video["num_layers"]:
        raise ValueError("interleaved readout requires equal action and video num_layers")
    if readout == "lite_kv_fusion" and action["num_layers"] > video["num_layers"]:
        raise ValueError("lite_kv_fusion requires action num_layers <= video num_layers")


def create_model(
    *,
    readout: str = "lite_kv_fusion",
    key_aggregation_order: str = "norm_rope_then_fuse",
    position_interface: str = "retracted_1d",
    action_text_cross_attn: bool = False,
    camera_layout: str = "horizontal",
    **model_kwargs: Any,
):
    """Construct Faster-WAM by default, or one of the six reported comparison arms.

    Action width and depth remain in ``action_dit_config``. The public TextCA
    flag takes precedence over its historical nested spelling. Unsupported
    combinations fail before backbone loading or checkpoint downloads.
    """
    if not isinstance(action_text_cross_attn, bool):
        raise TypeError("action_text_cross_attn must be a boolean")
    option_tuple = (readout, key_aggregation_order, position_interface, action_text_cross_attn)
    if option_tuple not in SUPPORTED_MODEL_OPTIONS:
        raise ValueError(
            f"Unsupported model options {option_tuple!r}; use one of the seven "
            "published model presets (readout, key_aggregation_order, "
            "position_interface, action_text_cross_attn)."
        )
    unexpected = sorted(set(model_kwargs) - _COMMON_ARGUMENTS)
    if unexpected:
        raise TypeError(f"Unsupported model arguments: {unexpected}")
    if camera_layout not in {"horizontal", "libero", "robotwin", "center"}:
        raise ValueError(f"Unknown camera_layout {camera_layout!r}")
    if position_interface == "lifted_3d" and camera_layout == "center":
        raise ValueError("lifted_3d requires a multi-camera layout")

    options = {
        "readout": readout,
        "key_aggregation_order": key_aggregation_order,
        "position_interface": position_interface,
        "action_text_cross_attn": action_text_cross_attn,
        "camera_layout": camera_layout,
    }
    kwargs = dict(model_kwargs)
    action_config = dict(kwargs.get("action_dit_config") or {})
    action_arguments = {
        "action_dim", "hidden_dim", "ffn_dim", "num_heads", "attn_head_dim",
        "num_layers", "text_dim", "freq_dim", "eps", "use_gradient_checkpointing",
        "action_text_cross_attn",
    }
    unsupported_action = sorted(set(action_config) - action_arguments)
    if unsupported_action:
        raise TypeError(f"Unsupported action_dit_config arguments: {unsupported_action}")
    _validate_dimensions(
        readout, dict(kwargs.get("video_dit_config") or {}), action_config,
    )
    action_config["action_text_cross_attn"] = action_text_cross_attn
    kwargs["action_dit_config"] = action_config
    # Explicit defaults also make the low-level historical factory signatures
    # consistent across all three readout strategies.
    for name, value in {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "tokenizer_max_len": 128,
        "load_text_encoder": False,
        "proprio_dim": None,
        "redirect_common_files": True,
        "mot_checkpoint_mixed_attn": True,
        "action_dit_pretrained_path": None,
        "skip_dit_load_from_pretrain": False,
    }.items():
        kwargs.setdefault(name, value)

    if readout == "interleaved":
        from fasterwam.runtime import create_fastwam
        model = create_fastwam(**kwargs)
    elif readout == "final_layer":
        from fasterwam.runtime import create_fastwam_decoupled
        model = create_fastwam_decoupled(
            **kwargs, decoupled=True, kv_source_mode="final_only",
        )
    else:
        from .wan22.dot import DockOfTransformers
        from .wan22.fasterwam_decoupled import create_fasterwam_decoupled
        freeze_video = kwargs.pop("freeze_video_backbone", False)
        rope_mode = (
            "original_3d" if key_aggregation_order == "fuse_then_norm_rope"
            else "aligned_3d" if position_interface == "lifted_3d"
            else "legacy_3d"
        )
        model = create_fasterwam_decoupled(
            **kwargs,
            decoupled=True,
            kv_source_mode="new_fused_kv",
            fixed_rope=True,
            new_fused_kv_rope_mode=rope_mode,
            aligned_3d_action_spatial_anchor_layout=camera_layout,
            new_fused_kv_projection_mode="simple_head_softmax",
            new_fused_kv_simple_head_softmax_fuse_mode="all",
            _mot_class=DockOfTransformers,
            _mot_options={
                "key_aggregation_order": key_aggregation_order,
                "position_interface": position_interface,
            },
        )
        model.freeze_video_backbone = bool(freeze_video)
    model.model_options = options
    return model
