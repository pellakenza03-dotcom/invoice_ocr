from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "invoice_summary.schema.json"


def valid_summary() -> dict:
    party = {
        "name": None,
        "address": None,
        "postal_code": None,
        "city": None,
        "country": None,
        "tax_id": None,
        "registration_id": None,
        "ice": None,
        "email": None,
        "phone": None,
    }
    return {
        "schema_version": "1.0.0",
        "document_id": "invoice-001",
        "document_type": "invoice",
        "invoice_number": "FAC-2026-001",
        "issue_date": "2026-09-11",
        "due_date": None,
        "currency": "EUR",
        "seller": {**party, "name": "Supplier SAS"},
        "buyer": {**party, "name": "Customer SARL"},
        "amounts": {
            "subtotal_excl_tax": 1000.0,
            "discount_amount": 0.0,
            "shipping_amount": None,
            "other_charges": None,
            "tax_amount": 200.0,
            "stamp_duty": None,
            "total_incl_tax": 1200.0,
            "deposit_amount": 300.0,
            "amount_paid": 0.0,
            "amount_due": 900.0,
        },
        "vat_breakdown": [
            {
                "rate": 20.0,
                "taxable_amount": 1000.0,
                "tax_amount": 200.0,
            }
        ],
        "provenance": [
            {
                "field_path": "/amounts/amount_due",
                "page_number": 1,
                "source_block_ids": ["p1_b31", "p1_b32"],
                "raw_text": "Net à payer 900,00 EUR",
                "ocr_confidence": 0.98,
            }
        ],
        "extraction_status": "complete",
        "warnings": [],
        "extraction_metadata": {
            "provider": "local",
            "model": "qwen",
            "prompt_version": "invoice-summary-v1",
            "generated_at": "2026-09-11T12:00:00Z",
            "duration_ms": 1450,
        },
    }


class InvoiceSummarySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(
            cls.schema, format_checker=FormatChecker()
        )

    def assert_valid(self, value: dict) -> None:
        errors = sorted(
            self.validator.iter_errors(value), key=lambda error: list(error.path)
        )
        self.assertEqual(errors, [], [error.message for error in errors])

    def assert_invalid(self, value: dict) -> None:
        self.assertTrue(list(self.validator.iter_errors(value)))

    def test_accepts_summary_with_nullable_fields(self):
        self.assert_valid(valid_summary())

    def test_rejects_missing_required_field(self):
        summary = valid_summary()
        del summary["buyer"]["name"]
        self.assert_invalid(summary)

    def test_rejects_line_items_and_other_unknown_fields(self):
        summary = valid_summary()
        summary["line_items"] = []
        self.assert_invalid(summary)

    def test_rejects_invalid_date_currency_and_confidence(self):
        invalid_values = []
        for path, value in (
            (("issue_date",), "11/09/2026"),
            (("currency",), "EURO"),
            (("provenance", 0, "ocr_confidence"), 1.1),
        ):
            summary = copy.deepcopy(valid_summary())
            target = summary
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            invalid_values.append(summary)

        for summary in invalid_values:
            with self.subTest(summary=summary):
                self.assert_invalid(summary)

    def test_rejects_invalid_ice_and_block_id(self):
        summary = valid_summary()
        summary["seller"]["ice"] = "123"
        self.assert_invalid(summary)

        summary = valid_summary()
        summary["provenance"][0]["source_block_ids"] = ["block-31"]
        self.assert_invalid(summary)


if __name__ == "__main__":
    unittest.main()
