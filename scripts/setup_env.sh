#!/usr/bin/env bash
# Install Faster-WAM and one benchmark into the active Python environment.
# Use a separate environment per target: train, libero, libero_plus and robotwin.
# Create it first with Python 3.10 and PyTorch 2.7.1 (see README.md).
set -euo pipefail
usage() {
  echo "Usage: bash scripts/setup_env.sh <train|libero|libero_plus|robotwin> [--dry-run]"
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# < 1 || $# > 2 )); then usage >&2; exit 2; fi
BENCHMARK="$1"
if [[ "${2:-}" == --dry-run ]]; then
  export DRY_RUN=1
elif (( $# == 2 )); then
  usage >&2; exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "${BENCHMARK}"
cd "${FASTERWAM_ROOT}"
PIP=("${PYTHON_BIN}" -m pip install)

# Pinned simulator stacks, with compatibility fixes described in the READMEs.
LIBERO_PACKAGES=(robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 future==0.18.2 easydict==1.9 matplotlib==3.5.3
  termcolor==2.5.0 opencv-python==4.11.0.86)
LIBERO_PLUS_PACKAGES=("usd-core>=25.5" wand==0.7.2 scikit-image==0.25.2)
LIBERO_PLUS_COMMIT=4976dc30028e805ff8094b55501d532c48fec182
CUROBO_TAG=v0.7.8

case "${BENCHMARK}" in
  train) ;;
  libero|libero_plus)
    if [[ ! -d "${LIBERO_ROOT:-}/libero/libero" ]]; then
      echo "Set LIBERO_ROOT (or LIBERO_PLUS_ROOT for Plus) to an existing benchmark checkout." >&2
      exit 2
    fi
    ;;
  robotwin)
    if [[ ! -d "${ROBOTWIN_ROOT}/envs" ]]; then
      echo "ROBOTWIN_ROOT must point to a RoboTwin checkout." >&2
      exit 2
    fi
    ;;
  *) usage >&2; exit 2 ;;
esac

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  if ! "${PYTHON_BIN}" -c 'import sys, torch; sys.exit(not (sys.version_info[:2] == (3, 10) and torch.__version__.startswith("2.7.1")))' 2>/dev/null; then
    echo "Activate an environment with Python 3.10 and PyTorch 2.7.1 first (see README.md)." >&2
    exit 2
  fi
  if [[ "${BENCHMARK}" == libero* ]]; then
    # LIBERO and LIBERO-Plus both install the `libero` package.
    INSTALLED=$(env -u PYTHONPATH "${PYTHON_BIN}" -c 'import libero, os; print(os.path.realpath(libero.__path__[0]))' 2>/dev/null || true)
    if [[ -n "${INSTALLED}" && "${INSTALLED}" != "$(realpath "${LIBERO_ROOT}")/libero" ]]; then
      echo "This environment already provides libero from ${INSTALLED}. Use a separate environment." >&2
      exit 2
    fi
  fi
fi

fasterwam_run "${PIP[@]}" -e "${FASTERWAM_ROOT}"

case "${BENCHMARK}" in
  libero|libero_plus)
    if [[ "${BENCHMARK}" == libero_plus ]]; then
      if [[ -d "${LIBERO_ROOT}/.git" ]] && [[ "$(git -C "${LIBERO_ROOT}" rev-parse HEAD)" != "${LIBERO_PLUS_COMMIT}" ]]; then
        echo "Warning: LIBERO-Plus is not at the tested commit ${LIBERO_PLUS_COMMIT}." >&2
      fi
      # The LIBERO-Plus checkout lacks the outer package marker needed by find_packages().
      if [[ ! -f "${LIBERO_ROOT}/libero/__init__.py" ]]; then
        fasterwam_run touch "${LIBERO_ROOT}/libero/__init__.py"
      fi
    fi
    fasterwam_run "${PIP[@]}" "${LIBERO_PACKAGES[@]}"
    fasterwam_run "${PIP[@]}" --no-deps robomimic==0.2.0
    if [[ "${BENCHMARK}" == libero_plus ]]; then
      fasterwam_run "${PIP[@]}" "${LIBERO_PLUS_PACKAGES[@]}"
    fi
    fasterwam_run "${PIP[@]}" mujoco==3.3.2 numpy==1.26.4
    # robosuite 1.4.0 hardcodes a shared /tmp file, which can belong to another
    # user on a multi-user machine and make even `import robosuite` fail.
    fasterwam_run "${PYTHON_BIN}" -c '
from importlib.metadata import distribution
path = distribution("robosuite").locate_file("robosuite/utils/log_utils.py")
text = path.read_text()
old = "logging.FileHandler(\"/tmp/robosuite.log\")"
new = "logging.FileHandler(os.path.join(tempfile.gettempdir(), f\"robosuite-{os.getuid()}.log\"))"
if old in text:
    text = text.replace("import logging\n", "import logging\nimport os\nimport tempfile\n", 1)
    path.write_text(text.replace(old, new))
    print(f"Patched {path}: per-user temporary log")
elif new not in text:
    raise RuntimeError(f"Unexpected robosuite logger source: {path}")
'
    fasterwam_run "${PIP[@]}" --no-deps -e "${LIBERO_ROOT}" --config-settings editable_mode=compat
    # PyTorch>=2.6 defaults torch.load to weights_only=True, which rejects the
    # benchmark's trusted init-state files. Patch the checkout once.
    fasterwam_run "${PYTHON_BIN}" -c '
import os
from pathlib import Path
path = Path(os.environ["LIBERO_ROOT"]) / "libero/libero/benchmark/__init__.py"
text = path.read_text()
old, new = "torch.load(init_states_path)", "torch.load(init_states_path, weights_only=False)"
if old in text:
    path.write_text(text.replace(old, new))
    print(f"Patched {path}: {new}")
'
    if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
      # YAML quoting also supports checkout paths containing spaces or colons.
      fasterwam_run "${PYTHON_BIN}" -c '
import os
from pathlib import Path
import yaml
root = Path(os.environ["LIBERO_ROOT"]).expanduser().resolve() / "libero"
path = Path(os.environ["LIBERO_CONFIG_PATH"]).expanduser() / "config.yaml"
values = {
    "benchmark_root": str(root / "libero"),
    "bddl_files": str(root / "libero/bddl_files"),
    "init_states": str(root / "libero/init_files"),
    "datasets": str(root / "datasets"),
    "assets": str(root / "libero/assets"),
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as handle:
    yaml.safe_dump(values, handle)
'
    fi
    if [[ "${DRY_RUN:-0}" != 1 && ! -d "${LIBERO_ROOT}/libero/libero/assets" ]]; then
      echo "Download the benchmark assets into ${LIBERO_ROOT}/libero/libero/assets (see README_Libero.md)." >&2
    fi
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      "${PYTHON_BIN}" -c 'from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; print("LIBERO simulator imports OK")'
    fi
    ;;
  robotwin)
    fasterwam_run "${PIP[@]}" -r "${ROBOTWIN_ROOT}/script/requirements.txt"
    CUROBO_DIR="${ROBOTWIN_ROOT}/envs/curobo"
    if [[ ! -d "${CUROBO_DIR}" ]]; then
      fasterwam_run git clone --branch "${CUROBO_TAG}" --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
    fi
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      # CuRobo compiles CUDA extensions; nvcc must match PyTorch's CUDA version.
      if [[ -z "${CUDA_HOME:-}" && -x "${CONDA_PREFIX:-}/bin/nvcc" ]]; then
        export CUDA_HOME="${CONDA_PREFIX}"
      fi
      NVCC="${CUDA_HOME:+${CUDA_HOME}/bin/}nvcc"
      TORCH_CUDA=$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda)')
      if ! "${NVCC}" --version 2>/dev/null | grep -q "release ${TORCH_CUDA}"; then
        echo "CuRobo needs nvcc ${TORCH_CUDA}. Install it into this environment and rerun:" >&2
        echo "  conda install -c nvidia cuda-nvcc=${TORCH_CUDA} cuda-cudart-dev=${TORCH_CUDA} -y" >&2
        exit 2
      fi
      export PATH="$(dirname "$(command -v "${NVCC}")"):${PATH}"
    fi
    fasterwam_run "${PIP[@]}" --no-build-isolation -e "${CUROBO_DIR}"
    # Warp 1.8.1 fixes CUDA initialization on NVIDIA R580 drivers while keeping
    # warp.torch, which CuRobo v0.7.8 needs and Warp 1.13 removed.
    # CuRobo's dependencies can upgrade these; mplib also needs NumPy 1.x.
    fasterwam_run "${PIP[@]}" warp-lang==1.8.1 numpy==1.26.4 setuptools==69.5.1
    if [[ ! -d "${ROBOTWIN_ROOT}/assets/embodiments" ]]; then
      fasterwam_run bash -c 'cd "$1" && bash script/_download_assets.sh' bash "${ROBOTWIN_ROOT}"
    fi
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      "${PYTHON_BIN}" -c '
import torch, sapien, mplib, curobo, warp
from curobo.types.math import Pose
warp.init()
if torch.cuda.is_available():
    if not warp.is_cuda_available():
        raise RuntimeError("PyTorch sees CUDA but Warp does not; check the Warp version and NVIDIA driver")
    warp.zeros(1, device="cuda")
    warp.synchronize()
    print("Warp CUDA initialization OK")
print("RoboTwin simulator packages OK")
'
    fi
    ;;
esac
echo "For subsequent commands: source scripts/env.sh ${BENCHMARK}"
