#!/usr/bin/env bash
# Runs presets sequentially. No scheduler submissions are made by this script.
# For eval all: ABLATION_CHECKPOINT_ROOT/<preset>.pt must exist for every preset.
set -euo pipefail
usage() {
  echo "Usage: bash scripts/ablation.sh <train|eval> <libero|libero_plus|robotwin> <preset|all> [gpus_per_node for train] [Hydra overrides...]"
  echo "Presets: fasterwam, fastwam, final_layer, fuse_before_norm_rope, fuse_after_norm_rope, no_textca, lifted_3d_rope"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# < 3 )); then usage >&2; exit 2; fi
MODE="$1"
BENCHMARK="$2"
PRESET="$3"
shift 3
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" train
cd "${FASTERWAM_ROOT}"
case "${MODE}" in
  train)
    if (( $# == 0 )); then usage >&2; exit 2; fi
    GPUS_PER_NODE="$1"; shift
    ;;
  eval) ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
case "${BENCHMARK}" in
  libero|libero_plus|robotwin) ;;
  *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac
PRESETS=(fasterwam fastwam final_layer fuse_before_norm_rope fuse_after_norm_rope no_textca lifted_3d_rope)
if [[ "${PRESET}" != all ]]; then
  case " ${PRESETS[*]} " in
    *" ${PRESET} "*) PRESETS=("${PRESET}") ;;
    *) echo "Unknown preset: ${PRESET}" >&2; exit 2 ;;
  esac
fi
# A sweep owns the model/output choices so every run remains distinct.
for arg in "$@"; do
  case "${arg}" in
    model=*|+model=*|++model=*|output_dir=*|+output_dir=*|++output_dir=*|EVALUATION.output_dir=*|+EVALUATION.output_dir=*|++EVALUATION.output_dir=*)
      echo "Use the preset argument and RUN_ID to select the model/output directory." >&2; exit 2 ;;
  esac
done
if [[ "${MODE}" == eval && "${PRESET}" == all ]]; then
  if [[ -z "${ABLATION_CHECKPOINT_ROOT:-}" ]]; then
    echo "eval all requires ABLATION_CHECKPOINT_ROOT containing <preset>.pt for all seven presets." >&2; exit 2
  fi
  for arg in "$@"; do
    case "${arg}" in
      ckpt=*|+ckpt=*|++ckpt=*) echo "eval all selects one checkpoint per preset; omit ckpt=." >&2; exit 2 ;;
    esac
  done
  # Validate the whole sweep before any evaluation starts.
  for model in "${PRESETS[@]}"; do
    if [[ ! -f "${ABLATION_CHECKPOINT_ROOT}/${model}.pt" ]]; then
      echo "Missing checkpoint: ${ABLATION_CHECKPOINT_ROOT}/${model}.pt" >&2; exit 2
    fi
  done
fi
if [[ "${MODE}" == train && " ${PRESETS[*]} " == *" fastwam "* && "${DRY_RUN:-0}" != 1 ]]; then
  ACTION_DIT_INIT="${ACTION_DIT_PRETRAINED_PATH:-./checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
  if [[ ! -f "${ACTION_DIT_INIT}" ]]; then
    echo "The fastwam preset needs its ActionDiT initialization: ${ACTION_DIT_INIT} (see README.md)." >&2; exit 2
  fi
fi
if (( ${NNODES:-1} > 1 )) && [[ -z "${RUN_ID:-}" ]]; then
  echo "Set the same RUN_ID on every node before multi-node training." >&2; exit 2
fi
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
for model in "${PRESETS[@]}"; do
  if [[ "${MODE}" == train ]]; then
    bash scripts/train.sh "${BENCHMARK}" "${GPUS_PER_NODE}" "$@" \
      "model=${model}" "output_dir=./runs/ablations/${BENCHMARK}/${RUN_ID}/${model}" \
      "wandb.name=${BENCHMARK}_${model}_${RUN_ID}"
  else
    CHECKPOINT_ARGS=()
    if [[ "${PRESET}" == all ]]; then
      CHECKPOINT_ARGS=("ckpt=${ABLATION_CHECKPOINT_ROOT}/${model}.pt")
    fi
    bash scripts/eval.sh "${BENCHMARK}" "$@" "${CHECKPOINT_ARGS[@]}" \
      "model=${model}" "EVALUATION.output_dir=./evaluate_results/ablations/${BENCHMARK}/${RUN_ID}/${model}"
  fi
done
