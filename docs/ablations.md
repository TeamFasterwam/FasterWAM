# Model presets and ablations

All presets are built by `fasterwam.models.factory.create_model` and selected with
`model=<preset>`. The default is `fasterwam`. The rows follow the ablation study
of the paper (Fig. 3, Sec. 3.3.2); each preset contains the complete architecture
settings of its row, so neighbouring rows can differ in more than one setting.

| Model (paper) | Preset (`model=...`) | Action readout | Key aggregation | Conditioning pathway | Positional interface | Action params |
| --- | --- | --- | --- | --- | --- | --- |
| Fast-WAM | `fastwam` | 30-layer interleaved MoT | Per-layer native KV | TextCA | Native (video 3D / action 1D) | 1.02B |
| Final-layer | `final_layer` | Final video layer | — | TextCA | Native | 46M |
| F → N+R | `fuse_before_norm_rope` | DoT (Lite KV-Fusion) | F → N+R | TextCA | Native | 46M |
| N+R → F | `fuse_after_norm_rope` | DoT (Lite KV-Fusion) | N+R → F | TextCA | Native | 46M |
| No TextCA | `no_textca` | DoT (Lite KV-Fusion) | N+R → F | World-model-only | Native | 29M |
| Lifted 3D RoPE | `lifted_3d_rope` | DoT (Lite KV-Fusion) | N+R → F | World-model-only | Lifted 3D RoPE | 29M |
| **Retracted 1D RoPE (Faster-WAM)** | `fasterwam` | DoT (Lite KV-Fusion) | N+R → F | World-model-only | Retracted 1D RoPE | 29M |

F is Lite KV-Fusion, a per-head softmax mixture over video layers; N is key
QK-Norm and R is rotary position embedding. F → N+R fuses keys before QK-Norm and
RoPE, while N+R → F fuses the backbone's normalized and rotated keys, reusing its
native KV cache. TextCA is the action expert's cross-attention to the language
and proprioception context. With world-model-only conditioning
(`action_text_cross_attn=false`) these inputs reach the action expert only
through the fused video representations; they still condition the video
backbone. `lifted_3d_rope` and `fasterwam` are the two positional alignments built
on `no_textca`. The model options are `readout` (`interleaved`, `final_layer`,
`lite_kv_fusion`), `key_aggregation_order` (`fuse_then_norm_rope`,
`norm_rope_then_fuse`), `position_interface` (`native_3d`, `lifted_3d`,
`retracted_1d`) and `action_text_cross_attn`.

The video backbone has 30 layers. The default action expert has one layer, hidden
dimension 1024, FFN dimension 4096 and 24 attention heads of dimension 128.
Lite KV-Fusion logits are initialized to zero (uniform layer weights). Action
experts are randomly initialized, except for the Fast-WAM baseline, which uses 30
action layers initialized from the interpolated ActionDiT checkpoint described in
[README.md](../README.md). The Fast-WAM/Final-layer comparison therefore also
changes initialization and depth.

## Positional interfaces

**Retracted 1D RoPE** (`fasterwam`). Lite KV-Fusion mixes the video keys after the
native QK-Norm and 3D RoPE. Retracted 1D-RoPE then removes the native 3D rotation
of the fused keys and re-encodes them on the action expert's 1D temporal axis and
frequency grid. Actions attend to the current video frame only (temporal position
0), so the re-encoding is the identity. There is no additional normalization of
the fused keys, and values, action queries and action keys are unchanged. In
reduced precision, rotating before or after fusion can give slightly different
results due to rounding; cached inference keeps the native video frequencies
needed for the retraction.

**Lifted 3D RoPE** (`lifted_3d_rope`). Action queries and keys use the video
backbone's temporal/height/width channel partition. Action tokens keep their
temporal index and receive spatial pseudo-anchors at the camera-view centers;
video keys keep their native 3D RoPE. Pseudo-anchors are interleaved across
spatial frequency pairs and heads: with two views (LIBERO) pair `r` of head `h`
uses view `(r + h) mod 2`; with three views (RoboTwin) the heads cycle through all
six permutations of (head camera, left wrist, right wrist), so each view takes 8
of the 24 heads at every frequency. The number of heads must be divisible by 6
for three views.

## Configuration examples

```bash
# Resolve a configuration without loading a model or data.
python scripts/train.py model=lifted_3d_rope --cfg job --resolve

# Train one ablation with the shared dataset and training settings.
bash scripts/ablation.sh train libero fuse_after_norm_rope 8 \
  gradient_accumulation_steps=2

# Change action-head depth and width.
bash scripts/train.sh libero 8 \
  model.action_dit_config.num_layers=3 \
  model.action_dit_config.hidden_dim=2048 \
  model.action_dit_config.ffn_dim=8192
```

`scripts/ablation.sh` accepts any preset in the table, or `all` to run all seven
sequentially, for `libero`, `libero_plus` or `robotwin`. LIBERO-Plus training uses
the LIBERO dataset and recipe. `all` checks before starting that the Fast-WAM
ActionDiT initialization exists (`ACTION_DIT_PRETRAINED_PATH`, default
`checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`).

```bash
# All seven presets, each with its own output directory.
RUN_ID=ablations bash scripts/ablation.sh train libero all 8 \
  gradient_accumulation_steps=2

# Evaluate one preset with its checkpoint.
bash scripts/ablation.sh eval libero_plus lifted_3d_rope \
  ckpt=/absolute/path/to/lifted_3d_rope.pt \
  EVALUATION.dataset_stats_path=/absolute/path/to/dataset_stats.json

# Evaluate all seven; the directory must contain <preset>.pt for every preset.
ABLATION_CHECKPOINT_ROOT=/absolute/path/to/checkpoints \
  bash scripts/ablation.sh eval robotwin all \
  EVALUATION.dataset_stats_path=/absolute/path/to/dataset_stats.json
```

Training outputs go to `runs/ablations/<benchmark>/<RUN_ID>/<preset>/` and
evaluation outputs to `evaluate_results/ablations/<benchmark>/<RUN_ID>/<preset>/`.
For `all` evaluation, name the checkpoints `fasterwam.pt`, `fastwam.pt`,
`final_layer.pt`, `fuse_before_norm_rope.pt`, `fuse_after_norm_rope.pt`,
`no_textca.pt` and `lifted_3d_rope.pt`. Share one statistics file
only when all checkpoints were trained on the same dataset recipe.

## Supported settings

Changing action width or depth changes the checkpoint shape. For Lite KV-Fusion,
action depth must be between 1 and the number of video layers (30). Interleaved
readout requires equal action and video depth; final-layer readout permits any
positive action depth. Action hidden and FFN widths are configurable, while the
attention head count and head dimension must match the video expert. Other
combinations of readout, aggregation and position interface raise an error
before any weights are downloaded. Faster-WAM disables gradient checkpointing
(`model.mot_checkpoint_mixed_attn=false`); enable it to trade compute for
activation memory.

## Checkpoints

A checkpoint stores the policy weights (`mot`, `proprio_encoder`), the training
step and the `model_options` of its preset (readout, key aggregation, positional
interface, TextCA and camera layout). Load each checkpoint with the preset and
dataset configuration it was trained with: presets with different positional
interfaces have identical parameter shapes, so a checkpoint whose
`model_options` differ from the configured model, or whose weights do not match
it one to one, is rejected. Full-state training resume uses
`resume=<run>/checkpoints/state/step_NNNNNN`.
