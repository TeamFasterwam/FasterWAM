#!/usr/bin/env bash
# Usage: source scripts/env.sh [train|libero|libero_plus|robotwin]
# Select the matching Python environment before sourcing this file.
export FASTERWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FASTERWAM_ROOT}/src:${FASTERWAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
# Wan2.2 weights are downloaded here on first use (several RoboTwin commands change directory).
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${FASTERWAM_ROOT}/checkpoints}"

case "${1:-train}" in
  train) ;;
  libero)
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    if [[ -n "${LIBERO_ROOT:-}" ]]; then
      export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"
    fi
    ;;
  libero_plus)
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero_plus}"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    if [[ -n "${LIBERO_PLUS_ROOT:-${LIBERO_ROOT:-}}" ]]; then
      export LIBERO_ROOT="${LIBERO_PLUS_ROOT:-${LIBERO_ROOT}}"
      export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"
    fi
    ;;
  robotwin)
    export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${FASTERWAM_ROOT}/third_party/RoboTwin}"
    ;;
  *) echo "Unknown environment: $1 (expected train, libero, libero_plus, robotwin)" >&2; return 2 ;;
esac

# Print argv with shell quoting, and optionally run it. No eval/string expansion.
fasterwam_run() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$@"
  fi
}
