"""End-to-end orchestration of semantic invoice-summary extraction."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context import OCRContext, build_ocr_context, load_ocr_document
from .llm import LLMBackend, LLMError, QwenTransformersLLM
from .normalize import NormalizationError, normalize_summary
from .prompt import PromptError, build_messages
from .validation import (
    SummaryValidationError,
    accounting_warnings,
    load_summary_schema,
    missing_core_field_warnings,
    missing_provenance_warnings,
    parse_json_response,
    validate_provenance,
    validate_summary_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "extraction.json"


class ExtractionError(RuntimeError):
    """Raised when the semantic extraction pipeline cannot produce a safe result."""


def _resolve_project_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionError(f"Configuration field {label} must be a path string.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_extraction_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    file = Path(path).expanduser().resolve()
    try:
        config = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExtractionError(f"Extraction configuration does not exist: {file}") from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Invalid extraction configuration {file}: {exc}") from exc
    if not isinstance(config, dict):
        raise ExtractionError("Extraction configuration must be a JSON object.")
    for key in ("model", "context", "validation"):
        if not isinstance(config.get(key), dict):
            raise ExtractionError(f"Configuration field {key} must be an object.")
    for key in ("ocr_schema_path", "summary_schema_path", "prompt_path"):
        config[key] = _resolve_project_path(config.get(key), key)
    offload_folder = config["model"].get("offload_folder")
    if offload_folder:
        config["model"]["offload_folder"] = str(
            _resolve_project_path(offload_folder, "model.offload_folder")
        )
    if not isinstance(config.get("prompt_version"), str) or not config["prompt_version"]:
        raise ExtractionError("Configuration field prompt_version must be a string.")
    return config


class InvoiceExtractionService:
    """Reusable service that loads Qwen once and extracts multiple invoices."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        backend: LLMBackend | None = None,
    ):
        self.config = load_extraction_config(config_path)
        self.summary_schema = load_summary_schema(self.config["summary_schema_path"])
        self.backend = backend

    def _backend(self) -> LLMBackend:
        if self.backend is None:
            backend_name = self.config["model"].get("backend")
            if backend_name != "transformers":
                raise ExtractionError(f"Unsupported LLM backend: {backend_name!r}")
            self.backend = QwenTransformersLLM(self.config["model"])
        return self.backend

    def prepare_context(self, document: dict[str, Any]) -> OCRContext:
        settings = self.config["context"]
        return build_ocr_context(
            document,
            coordinate_scale=int(settings.get("coordinate_scale", 1000)),
            max_characters=settings.get("max_characters"),
        )

    def load_and_prepare_context(self, path: str | Path) -> OCRContext:
        document = load_ocr_document(path, self.config["ocr_schema_path"])
        return self.prepare_context(document)

    def extract_document(self, document: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        context = self.prepare_context(document)
        messages = build_messages(
            context,
            prompt_path=self.config["prompt_path"],
            summary_schema_path=self.config["summary_schema_path"],
        )
        backend = self._backend()
        try:
            raw_response = backend.generate(messages)
            summary = normalize_summary(parse_json_response(raw_response))
        except (LLMError, NormalizationError, SummaryValidationError) as exc:
            raise ExtractionError(str(exc)) from exc

        duration_ms = round((time.perf_counter() - started) * 1000)
        summary["schema_version"] = "1.0.0"
        summary["document_id"] = document["document_id"]
        summary["extraction_metadata"] = {
            "provider": backend.provider,
            "model": backend.model_id,
            "prompt_version": self.config["prompt_version"],
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration_ms": duration_ms,
        }
        try:
            validate_summary_schema(summary, self.summary_schema)
            summary["warnings"] = list(dict.fromkeys(summary["warnings"]))
            validate_provenance(
                summary,
                document,
                allowed_block_ids=context.included_block_ids,
            )
        except SummaryValidationError as exc:
            raise ExtractionError(str(exc)) from exc

        review_warnings = accounting_warnings(
            summary,
            tolerance=float(self.config["validation"].get("amount_tolerance", 0.02)),
        )
        review_warnings.extend(missing_provenance_warnings(summary))
        if context.truncated:
            review_warnings.append(
                f"OCR context was truncated; {len(context.omitted_block_ids)} block(s) "
                "were not sent to the model."
            )
        partial_warnings = missing_core_field_warnings(summary)
        summary["warnings"] = list(
            dict.fromkeys(summary["warnings"] + review_warnings + partial_warnings)
        )
        if review_warnings or summary.get("extraction_status") == "review_required":
            summary["extraction_status"] = "review_required"
        elif partial_warnings:
            summary["extraction_status"] = "partial"
        else:
            summary["extraction_status"] = "complete"
        validate_summary_schema(summary, self.summary_schema)
        return summary

    def extract_file(self, path: str | Path) -> dict[str, Any]:
        document = load_ocr_document(path, self.config["ocr_schema_path"])
        return self.extract_document(document)
