"""Deterministic normalization for values emitted by the language model."""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime
from typing import Any


AMOUNT_FIELDS = {
    "subtotal_excl_tax",
    "discount_amount",
    "shipping_amount",
    "other_charges",
    "tax_amount",
    "stamp_duty",
    "total_incl_tax",
    "deposit_amount",
    "amount_paid",
    "amount_due",
}
CURRENCY_ALIASES = {
    "€": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "$": "USD",
    "US$": "USD",
    "USD": "USD",
    "DH": "MAD",
    "DHS": "MAD",
    "MAD": "MAD",
}
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")


class NormalizationError(ValueError):
    """Raised when a non-null business value cannot be normalized safely."""


def normalize_string(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalize_currency(value: object) -> str | None:
    text = normalize_string(value)
    if text is None:
        return None
    upper = text.upper().replace(".", "")
    return CURRENCY_ALIASES.get(upper, upper)


def normalize_date(value: object) -> str | None:
    text = normalize_string(value)
    if text is None:
        return None
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    raise NormalizationError(f"Cannot normalize date: {text!r}")


def normalize_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise NormalizationError(f"Boolean is not a monetary value: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise NormalizationError(f"Non-finite monetary value: {value!r}")
        return number

    text = normalize_string(value)
    if text is None:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9,.'+-]", "", text)
    cleaned = cleaned.replace("'", "")
    if not cleaned or not re.search(r"[0-9]", cleaned):
        raise NormalizationError(f"Cannot normalize number: {text!r}")

    sign = -1 if negative or cleaned.startswith("-") else 1
    cleaned = cleaned.lstrip("+-")
    if "," in cleaned and "." in cleaned:
        decimal_separator = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        cleaned = cleaned.replace(thousands_separator, "")
        cleaned = cleaned.replace(decimal_separator, ".")
    elif "," in cleaned or "." in cleaned:
        separator = "," if "," in cleaned else "."
        parts = cleaned.split(separator)
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            cleaned = "".join(parts)
        else:
            cleaned = ".".join(parts)
    try:
        return sign * float(cleaned)
    except ValueError as exc:
        raise NormalizationError(f"Cannot normalize number: {text!r}") from exc


def normalize_summary(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize known summary fields without inventing missing values."""

    result = copy.deepcopy(value)
    for key in ("invoice_number",):
        if key in result:
            result[key] = normalize_string(result[key])
    for key in ("issue_date", "due_date"):
        if key in result:
            result[key] = normalize_date(result[key])
    if "currency" in result:
        result["currency"] = normalize_currency(result["currency"])

    for party_key in ("seller", "buyer"):
        party = result.get(party_key)
        if isinstance(party, dict):
            for key, item in party.items():
                party[key] = normalize_string(item)
            if isinstance(party.get("email"), str):
                party["email"] = party["email"].lower()

    amounts = result.get("amounts")
    if isinstance(amounts, dict):
        for key in AMOUNT_FIELDS & amounts.keys():
            amounts[key] = normalize_number(amounts[key])

    vat_breakdown = result.get("vat_breakdown")
    if isinstance(vat_breakdown, list):
        for line in vat_breakdown:
            if not isinstance(line, dict):
                continue
            for key in ("rate", "taxable_amount", "tax_amount"):
                if key in line:
                    line[key] = normalize_number(line[key])
    return result
