"""Schema, provenance and accounting validation for invoice summaries."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker


class SummaryValidationError(ValueError):
    """Raised when an extracted invoice summary is not trustworthy."""


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse exactly one JSON object, allowing only a surrounding Markdown fence."""

    text = response.strip()
    text = re.sub(r"^<think>[\s\S]*?</think>\s*", "", text, count=1)
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SummaryValidationError(
            f"The model did not return exactly one valid JSON object: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SummaryValidationError("The model response must be a JSON object.")
    return value


def load_summary_schema(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    try:
        schema = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SummaryValidationError(f"Summary schema does not exist: {file}") from exc
    except json.JSONDecodeError as exc:
        raise SummaryValidationError(f"Invalid summary schema {file}: {exc}") from exc
    Draft202012Validator.check_schema(schema)
    return schema


def validate_summary_schema(
    summary: dict[str, Any], schema: dict[str, Any]
) -> None:
    errors = sorted(
        Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(summary),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"/{'/'.join(map(str, error.absolute_path))}: {error.message}"
            for error in errors[:10]
        )
        raise SummaryValidationError(f"Invoice summary validation failed: {details}")


def _resolve_pointer(document: object, pointer: str) -> object:
    current = document
    for raw_part in pointer.lstrip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise KeyError(part)
        except (KeyError, IndexError, ValueError) as exc:
            raise SummaryValidationError(
                f"Provenance field_path does not exist: {pointer}"
            ) from exc
    return current


def _compact_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def validate_provenance(
    summary: dict[str, Any],
    ocr_document: dict[str, Any],
    *,
    allowed_block_ids: Iterable[str] | None = None,
) -> None:
    """Ensure every cited proof refers to actual OCR supplied to the model."""

    block_index: dict[str, tuple[int, dict[str, Any]]] = {}
    for page in ocr_document["pages"]:
        for block in page["blocks"]:
            block_index[block["block_id"]] = (page["page_number"], block)
    allowed = set(allowed_block_ids) if allowed_block_ids is not None else None

    for evidence in summary["provenance"]:
        pointer = evidence["field_path"]
        if _resolve_pointer(summary, pointer) is None:
            raise SummaryValidationError(
                f"Provenance cannot target a null value: {pointer}"
            )
        cited_blocks: list[dict[str, Any]] = []
        for block_id in evidence["source_block_ids"]:
            if block_id not in block_index:
                raise SummaryValidationError(
                    f"Provenance references unknown OCR block: {block_id}"
                )
            if allowed is not None and block_id not in allowed:
                raise SummaryValidationError(
                    f"Provenance references an OCR block not sent to the model: {block_id}"
                )
            page_number, block = block_index[block_id]
            if page_number != evidence["page_number"]:
                raise SummaryValidationError(
                    f"Provenance page mismatch for OCR block {block_id}."
                )
            cited_blocks.append(block)

        cited_text = _compact_text(
            " ".join(str(block.get("text", "")) for block in cited_blocks)
        )
        raw_text = _compact_text(evidence["raw_text"])
        if raw_text not in cited_text and cited_text not in raw_text:
            raise SummaryValidationError(
                f"Provenance raw_text does not match cited OCR for {pointer}."
            )


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return Decimal(str(value))


def accounting_warnings(
    summary: dict[str, Any], *, tolerance: float = 0.02
) -> list[str]:
    """Return deterministic warnings without changing extracted monetary values."""

    totals = summary["amounts"]
    allowed_difference = Decimal(str(tolerance))
    warnings: list[str] = []

    subtotal = _decimal(totals.get("subtotal_excl_tax"))
    tax = _decimal(totals.get("tax_amount"))
    total = _decimal(totals.get("total_incl_tax"))
    if subtotal is not None and tax is not None and total is not None:
        additions = sum(
            (
                _decimal(totals.get(key)) or Decimal("0")
                for key in ("shipping_amount", "other_charges", "stamp_duty")
            ),
            Decimal("0"),
        )
        expected_total = subtotal + tax + additions
        if abs(expected_total - total) > allowed_difference:
            warnings.append(
                "Total HT, taxes and charges do not reconcile with total TTC."
            )

    deposit = _decimal(totals.get("deposit_amount"))
    paid = _decimal(totals.get("amount_paid"))
    due = _decimal(totals.get("amount_due"))
    if None not in (total, deposit, paid, due):
        expected_due = total - deposit - paid
        if abs(expected_due - due) > allowed_difference:
            warnings.append(
                "Total TTC, deposit and paid amount do not reconcile with amount due."
            )
    return warnings


def missing_core_field_warnings(summary: dict[str, Any]) -> list[str]:
    fields = {
        "/invoice_number": summary.get("invoice_number"),
        "/seller/name": summary.get("seller", {}).get("name"),
        "/buyer/name": summary.get("buyer", {}).get("name"),
    }
    totals = summary.get("amounts", {})
    if totals.get("amount_due") is None and totals.get("total_incl_tax") is None:
        fields["/amounts/amount_due or /amounts/total_incl_tax"] = None
    return [f"Core field was not extracted: {path}." for path, value in fields.items() if value is None]


def missing_provenance_warnings(summary: dict[str, Any]) -> list[str]:
    """Report important extracted fields for which the model supplied no OCR proof."""

    candidates = {
        "/invoice_number": summary.get("invoice_number"),
        "/issue_date": summary.get("issue_date"),
        "/due_date": summary.get("due_date"),
        "/currency": summary.get("currency"),
        "/seller/name": summary.get("seller", {}).get("name"),
        "/buyer/name": summary.get("buyer", {}).get("name"),
    }
    candidates.update(
        {
            f"/amounts/{key}": value
            for key, value in summary.get("amounts", {}).items()
        }
    )
    proven = {item["field_path"] for item in summary.get("provenance", [])}
    return [
        f"Extracted field has no OCR provenance: {path}."
        for path, value in candidates.items()
        if value is not None and path not in proven
    ]
