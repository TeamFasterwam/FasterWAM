"""Encode standard LIBERO task prompts with the shared text encoder at batch size one."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from fasterwam.tools import precompute_text_embeds


STANDARD_LIBERO_PRECOMPUTE_BATCH_SIZE = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compose_precompute_config(overrides: list[str]) -> DictConfig:
    """Compose the training configuration from the absolute repository config directory."""
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "configs"),
    ):
        return compose(config_name="train", overrides=overrides)


def main() -> None:
    """Invoke the shared Hydra entrypoint after selecting batch size one."""
    cfg = compose_precompute_config(sys.argv[1:])
    precompute_text_embeds.DEFAULT_BATCH_SIZE = STANDARD_LIBERO_PRECOMPUTE_BATCH_SIZE
    precompute_text_embeds.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
