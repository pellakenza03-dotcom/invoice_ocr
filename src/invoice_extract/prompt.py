"""Prompt construction for invoice-summary extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .context import OCRContext


class PromptError(ValueError):
    """Raised when prompt resources cannot be loaded."""


def _read_text(path: str | Path) -> str:
    file = Path(path)
    try:
        value = file.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise PromptError(f"Prompt file does not exist: {file}") from exc
    if not value:
        raise PromptError(f"Prompt file is empty: {file}")
    return value


def _read_schema(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromptError(f"Invoice summary schema does not exist: {file}") from exc
    except json.JSONDecodeError as exc:
        raise PromptError(f"Invalid invoice summary schema {file}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromptError(f"Invoice summary schema must be an object: {file}")
    return value


def build_messages(
    context: OCRContext,
    *,
    prompt_path: str | Path,
    summary_schema_path: str | Path,
) -> list[dict[str, str]]:
    """Build chat messages containing rules, schema and untrusted OCR evidence."""

    system_prompt = _read_text(prompt_path)
    schema = _read_schema(summary_schema_path)
    user_content = (
        "Return one JSON object matching this JSON Schema:\n"
        "<output_schema>\n"
        f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}\n"
        "</output_schema>\n\n"
        "Extract values only from this OCR context:\n"
        "<ocr_context>\n"
        f"{context.text}\n"
        "</ocr_context>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
