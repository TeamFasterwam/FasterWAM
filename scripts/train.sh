#!/usr/bin/env bash
# One process per GPU; for multiple nodes set NNODES, NODE_RANK, MASTER_ADDR,
# MASTER_PORT and the same RUN_ID on every node. Slurm sets these automatically.
set -euo pipefail
usage() {
  echo "Usage: bash scripts/train.sh <libero|libero_plus|robotwin> <gpus_per_node> [Hydra overrides...]"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# < 2 )); then usage >&2; exit 2; fi
BENCHMARK="$1"
GPUS_PER_NODE="$2"
shift 2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" train
cd "${FASTERWAM_ROOT}"
case "${BENCHMARK}" in
  libero|libero_plus) DATA_CONFIG=libero_2cam ;;
  robotwin) DATA_CONFIG=robotwin ;;
  *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
if [[ ! "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ || ! "${NNODES}" =~ ^[1-9][0-9]*$ || ! "${NODE_RANK}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "GPU/node counts must be positive integers and NODE_RANK a nonnegative integer." >&2
  exit 2
fi
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be less than NNODES." >&2; exit 2
fi
if (( NNODES > 1 )) && [[ -z "${RUN_ID:-}" ]]; then
  echo "Set the same RUN_ID on every node before multi-node training." >&2; exit 2
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ACCELERATE_ARGS=(
  --config_file configs/distributed/zero1.yaml
  --num_processes "$((GPUS_PER_NODE * NNODES))"
  --num_machines "${NNODES}"
  --machine_rank "${NODE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
)
if (( NNODES > 1 )); then
  ACCELERATE_ARGS+=(--deepspeed_multinode_launcher standard)
fi
fasterwam_run "${ACCELERATE_BIN}" launch "${ACCELERATE_ARGS[@]}" scripts/train.py \
  "data=${DATA_CONFIG}" "output_dir=./runs/${BENCHMARK}/${RUN_ID}" \
  "wandb.name=${BENCHMARK}_${RUN_ID}" "$@"
