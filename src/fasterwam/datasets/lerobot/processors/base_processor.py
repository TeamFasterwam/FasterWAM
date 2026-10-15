"""Interface between LeRobot datasets and model-specific preprocessing."""

from abc import ABC, abstractmethod
from typing import Any

from ..utils.normalizer import LinearNormalizer


class BaseProcessor(ABC):
    @property
    @abstractmethod
    def is_train(self) -> bool:
        """Whether training transforms are enabled."""

    @property
    @abstractmethod
    def normalizer(self) -> LinearNormalizer:
        """Return the initialized action/state normalizer."""

    @abstractmethod
    def train(self) -> "BaseProcessor":
        """Select training transforms and return this processor."""

    @abstractmethod
    def eval(self) -> "BaseProcessor":
        """Select evaluation transforms and return this processor."""

    @abstractmethod
    def set_normalizer_from_stats(self, dataset_stats: dict[str, Any] | None = None) -> None:
        """Initialize normalization from dataset statistics."""

    @abstractmethod
    def action_state_transform(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Transform raw actions and states before normalization."""

    @abstractmethod
    def preprocess_images(self, images: dict[str, Any], **kwargs: Any) -> Any:
        """Process camera streams for training or VAE precomputation."""

    @abstractmethod
    def preprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert a dataset sample into model inputs."""

    @abstractmethod
    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert model outputs back into dataset action coordinates."""
