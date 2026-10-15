#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: bash scripts/eval.sh <libero|libero_plus|robotwin> ckpt=<path> [Hydra overrides...]"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# < 1 )); then usage >&2; exit 2; fi
BENCHMARK="$1"
shift
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "${BENCHMARK}"
cd "${FASTERWAM_ROOT}"
case "${BENCHMARK}" in
  libero)
    fasterwam_run "${PYTHON_BIN}" -m experiments.libero.run_libero_manager_v2 "$@"
    ;;
  libero_plus)
    fasterwam_run "${PYTHON_BIN}" -m experiments.libero.run_libero_manager_v2 \
      --config-name=sim_libero_plus "$@"
    ;;
  robotwin)
    fasterwam_run "${PYTHON_BIN}" -m experiments.robotwin.run_robotwin_manager "$@"
    ;;
  *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 2 ;;
esac
