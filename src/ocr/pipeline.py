"""Persistent PP-StructureV3 pipeline used by the OCR worker.
``PaddleOCRService`` owns that long-lived model
instance and serializes inference because this project targets a CPU-only host.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ocr.json"
SUPPORTED_INPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class OCRConfigurationError(ValueError):
    """Raised when the OCR configuration is invalid."""


class OCRInputError(ValueError):
    """Raised when a document cannot be submitted to OCR."""


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Typed representation of values whose single source is ``ocr.json``."""

    device: str
    text_detection_model_name: str
    text_recognition_model_name: str
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool
    use_textline_orientation: bool
    use_seal_recognition: bool
    use_table_recognition: bool
    use_formula_recognition: bool
    use_chart_recognition: bool
    use_region_detection: bool
    enable_mkldnn: bool
    cpu_threads: int

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise OCRConfigurationError(
                "This deployment is configured for CPU only; device must be 'cpu'."
            )
        if self.cpu_threads < 1:
            raise OCRConfigurationError("cpu_threads must be greater than zero.")

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "OCRConfig":
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise OCRConfigurationError(
                f"OCR configuration file not found: {config_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OCRConfigurationError(
                f"Invalid JSON in OCR configuration {config_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise OCRConfigurationError("OCR configuration must be a JSON object.")

        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise OCRConfigurationError(
                f"Unknown OCR configuration field(s): {', '.join(unknown)}"
            )

        try:
            return cls(**raw)
        except TypeError as exc:
            raise OCRConfigurationError(f"Invalid OCR configuration: {exc}") from exc

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        return asdict(self)


class PaddleOCRService:
    """Load PP-StructureV3 once and reuse it for every submitted document."""

    def __init__(
        self,
        config: OCRConfig | None = None,
        *,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or OCRConfig.from_json()
        self._inference_lock = Lock()

        if pipeline_factory is None:
            # Keep the costly Paddle import and model initialization out of module import.
            from paddleocr import PPStructureV3

            pipeline_factory = PPStructureV3

        self._pipeline = pipeline_factory(**self.config.to_pipeline_kwargs())

    def predict(self, document_path: str | Path) -> list[Any]:
        """Run PP-StructureV3 and return one native Paddle result per page."""

        path = Path(document_path).expanduser().resolve()
        if not path.is_file():
            raise OCRInputError(f"OCR input file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
            raise OCRInputError(
                f"Unsupported OCR input type '{path.suffix}'. Supported: {supported}"
            )

        # A single inference at a time avoids RAM spikes on the 16 GB CPU host.
        with self._inference_lock:
            results = self._pipeline.predict(str(path))

        return list(results)


@lru_cache(maxsize=1)
def get_ocr_service(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PaddleOCRService:
    """Return the process-wide OCR service singleton."""

    return PaddleOCRService(OCRConfig.from_json(config_path))
