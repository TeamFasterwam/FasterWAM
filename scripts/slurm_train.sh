#!/usr/bin/env bash
#SBATCH --job-name=fasterwam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Activate the training environment before sbatch. Add --partition, --qos or --account
# for your cluster on the sbatch command line; resource flags can be overridden there.
# Usage: sbatch scripts/slurm_train.sh <libero|libero_plus|robotwin> [Hydra overrides...]
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: sbatch scripts/slurm_train.sh <libero|libero_plus|robotwin> [Hydra overrides...]" >&2
  exit 2
fi
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
source scripts/env.sh train
export NNODES="${SLURM_NNODES:-1}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
else
  mapfile -t JOB_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  export MASTER_ADDR="${MASTER_ADDR:-${JOB_HOSTS[0]}}"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID:-preview}}"
fasterwam_run srun --kill-on-bad-exit=1 --ntasks="${NNODES}" --ntasks-per-node=1 bash -c '
  export NODE_RANK="${SLURM_NODEID}"
  benchmark="$1"
  shift
  exec bash scripts/train.sh "${benchmark}" "${GPUS_PER_NODE}" "$@"
' bash "$@"
