"""Offline preparation tools used by the training and evaluation workflows."""

import os
from pathlib import Path


def config_dir() -> str:
    """Locate source-checkout configs or an explicitly supplied config directory."""
    source_configs = Path(__file__).resolve().parents[3] / "configs"
    default = source_configs if source_configs.is_dir() else Path.cwd() / "configs"
    return str(Path(os.environ.get("FASTERWAM_CONFIG_DIR", default)).expanduser().resolve())
