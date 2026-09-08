"""Persistent PP-StructureV3 pipeline used by the OCR worker.
``PaddleOCRService`` owns that long-lived model
instance and serializes inference because this project targets a CPU-only host.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ocr.json"
SUPPORTED_INPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PREDICT_ONLY_FIELDS = (
    "use_table_orientation_classify",
    "use_ocr_results_with_table_cells",
    "use_e2e_wired_table_rec_model",
    "use_e2e_wireless_table_rec_model",
    "use_wired_table_cells_trans_to_html",
    "use_wireless_table_cells_trans_to_html",
)


class OCRConfigurationError(ValueError):
    """Raised when the OCR configuration is invalid."""


class OCRInputError(ValueError):
    """Raised when a document cannot be submitted to OCR."""


@dataclass(frozen=True, slots=True)
class OCRConfig:
    """Typed representation of values whose single source is ``ocr.json``."""

    device: str
    layout_detection_model_name: str
    text_detection_model_name: str
    text_recognition_model_name: str
    table_classification_model_name: str
    wired_table_structure_recognition_model_name: str
    wireless_table_structure_recognition_model_name: str
    wired_table_cells_detection_model_name: str
    wireless_table_cells_detection_model_name: str
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool
    use_textline_orientation: bool
    use_seal_recognition: bool
    use_table_recognition: bool
    use_table_orientation_classify: bool
    use_ocr_results_with_table_cells: bool
    use_e2e_wired_table_rec_model: bool
    use_e2e_wireless_table_rec_model: bool
    use_wired_table_cells_trans_to_html: bool
    use_wireless_table_cells_trans_to_html: bool
    use_formula_recognition: bool
    use_chart_recognition: bool
    use_region_detection: bool
    layout_threshold: float
    layout_nms: bool
    layout_merge_bboxes_mode: str
    enable_mkldnn: bool
    cpu_threads: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"(?:cpu|gpu(?::\d+)?)", self.device):
            raise OCRConfigurationError(
                "device must be 'cpu', 'gpu', or an indexed GPU such as 'gpu:0'."
            )
        if self.cpu_threads < 1:
            raise OCRConfigurationError("cpu_threads must be greater than zero.")
        if not 0 <= self.layout_threshold <= 1:
            raise OCRConfigurationError("layout_threshold must be between 0 and 1.")
        if not isinstance(self.layout_nms, bool):
            raise OCRConfigurationError("layout_nms must be a boolean.")
        if self.layout_merge_bboxes_mode not in {"large", "small", "union"}:
            raise OCRConfigurationError(
                "layout_merge_bboxes_mode must be 'large', 'small', or 'union'."
            )
        expected_models = {
            "layout_detection_model_name": "PP-DocLayout_plus-L",
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "latin_PP-OCRv5_mobile_rec",
            "table_classification_model_name": "PP-LCNet_x1_0_table_cls",
            "wired_table_structure_recognition_model_name": "SLANeXt_wired",
            "wireless_table_structure_recognition_model_name": "SLANet_plus",
            "wired_table_cells_detection_model_name": (
                "RT-DETR-L_wired_table_cell_det"
            ),
            "wireless_table_cells_detection_model_name": (
                "RT-DETR-L_wireless_table_cell_det"
            ),
        }
        for field_name, expected in expected_models.items():
            if getattr(self, field_name) != expected:
                raise OCRConfigurationError(
                    f"{field_name} must be {expected!r} for this pipeline."
                )
        boolean_fields = (
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
            "use_seal_recognition",
            "use_table_recognition",
            "use_table_orientation_classify",
            "use_ocr_results_with_table_cells",
            "use_e2e_wired_table_rec_model",
            "use_e2e_wireless_table_rec_model",
            "use_wired_table_cells_trans_to_html",
            "use_wireless_table_cells_trans_to_html",
            "use_formula_recognition",
            "use_chart_recognition",
            "use_region_detection",
            "enable_mkldnn",
        )
        for field_name in boolean_fields:
            if not isinstance(getattr(self, field_name), bool):
                raise OCRConfigurationError(f"{field_name} must be a boolean.")

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
        values = asdict(self)
        for field_name in PREDICT_ONLY_FIELDS:
            values.pop(field_name)
        return values

    def to_predict_kwargs(self) -> dict[str, bool]:
        """Return table-routing options accepted only by ``predict()``."""

        values = asdict(self)
        return {field_name: values[field_name] for field_name in PREDICT_ONLY_FIELDS}

    def with_overrides(self, overrides: dict[str, Any]) -> "OCRConfig":
        """Return a validated copy with values overridden by an audit profile."""

        allowed = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise OCRConfigurationError(
                f"Unknown OCR override field(s): {', '.join(unknown)}"
            )
        try:
            return replace(self, **overrides)
        except TypeError as exc:
            raise OCRConfigurationError(f"Invalid OCR overrides: {exc}") from exc


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
            results = self._pipeline.predict(
                str(path),
                **self.config.to_predict_kwargs(),
            )

        return list(results)


@lru_cache(maxsize=1)
def get_ocr_service(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PaddleOCRService:
    """Return the process-wide OCR service singleton."""

    return PaddleOCRService(OCRConfig.from_json(config_path))
