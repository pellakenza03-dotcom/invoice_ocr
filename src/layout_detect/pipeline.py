"""Persistent, CPU-only inference for the standalone layout model."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCR_CONFIG_PATH = PROJECT_ROOT / "config" / "ocr.json"
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


class LayoutConfigurationError(ValueError):
    """Raised when the layout configuration is missing or invalid."""


class LayoutInputError(ValueError):
    """Raised when an input cannot be sent to the layout model."""


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """Layout-only view over the project's shared OCR configuration."""

    model_name: str
    device: str
    threshold: float
    layout_nms: bool
    layout_merge_bboxes_mode: str
    enable_mkldnn: bool
    cpu_threads: int

    def __post_init__(self) -> None:
        if self.model_name != "PP-DocLayout_plus-L":
            raise LayoutConfigurationError(
                "layout_detection_model_name must be 'PP-DocLayout_plus-L'."
            )
        if self.device != "cpu":
            raise LayoutConfigurationError(
                "This local command is CPU-only; device must be 'cpu'."
            )
        if not 0 <= self.threshold <= 1:
            raise LayoutConfigurationError("Layout threshold must be between 0 and 1.")
        if not isinstance(self.layout_nms, bool):
            raise LayoutConfigurationError("layout_nms must be a boolean.")
        if self.layout_merge_bboxes_mode not in {"large", "small", "union"}:
            raise LayoutConfigurationError(
                "layout_merge_bboxes_mode must be 'large', 'small', or 'union'."
            )
        if self.cpu_threads < 1:
            raise LayoutConfigurationError("cpu_threads must be greater than zero.")

    @classmethod
    def from_ocr_json(
        cls, path: str | Path = DEFAULT_OCR_CONFIG_PATH
    ) -> "LayoutConfig":
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LayoutConfigurationError(
                f"Shared OCR configuration not found: {config_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LayoutConfigurationError(
                f"Invalid JSON in shared OCR configuration: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise LayoutConfigurationError(
                "Shared OCR configuration must be a JSON object."
            )

        required_mapping = {
            "model_name": "layout_detection_model_name",
            "device": "device",
            "threshold": "layout_threshold",
            "layout_nms": "layout_nms",
            "layout_merge_bboxes_mode": "layout_merge_bboxes_mode",
            "enable_mkldnn": "enable_mkldnn",
            "cpu_threads": "cpu_threads",
        }
        missing = sorted(
            source_name
            for source_name in required_mapping.values()
            if source_name not in raw
        )
        if missing:
            raise LayoutConfigurationError(
                f"Missing shared configuration field(s): {', '.join(missing)}"
            )
        return cls(
            **{
                target_name: raw[source_name]
                for target_name, source_name in required_mapping.items()
            }
        )

    def with_profile(self, overrides: dict[str, object]) -> "LayoutConfig":
        """Translate the shared audit profile into standalone model options."""

        mapping = {
            "layout_threshold": "threshold",
            "layout_nms": "layout_nms",
            "layout_merge_bboxes_mode": "layout_merge_bboxes_mode",
        }
        unknown = sorted(set(overrides) - set(mapping))
        if unknown:
            raise LayoutConfigurationError(
                f"Unsupported layout profile field(s): {', '.join(unknown)}"
            )
        translated = {mapping[name]: value for name, value in overrides.items()}
        try:
            return replace(self, **translated)
        except TypeError as exc:
            raise LayoutConfigurationError(f"Invalid layout profile: {exc}") from exc

    def to_model_kwargs(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "threshold": self.threshold,
            "layout_nms": self.layout_nms,
            "layout_merge_bboxes_mode": self.layout_merge_bboxes_mode,
            "enable_mkldnn": self.enable_mkldnn,
            "cpu_threads": self.cpu_threads,
        }


class LayoutDetectionService:
    """Load only PP-DocLayout_plus-L once and serialize CPU inference."""

    def __init__(
        self,
        config: LayoutConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._inference_lock = Lock()
        if model_factory is None:
            from paddleocr import LayoutDetection

            model_factory = LayoutDetection
        self._model = model_factory(**config.to_model_kwargs())

    def predict(self, image_path: str | Path) -> list[Any]:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise LayoutInputError(f"Layout input image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
            raise LayoutInputError(
                f"Unsupported layout input '{path.suffix}'. Supported: {supported}"
            )
        with self._inference_lock:
            return list(self._model.predict(str(path)))
