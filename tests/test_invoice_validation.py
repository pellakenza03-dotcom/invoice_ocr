from __future__ import annotations

import json
import unittest

from invoice_extract.validation import (
    SummaryValidationError,
    accounting_warnings,
    parse_json_response,
    validate_provenance,
)


class InvoiceValidationTests(unittest.TestCase):
    def test_parses_plain_or_fenced_json_but_rejects_extra_prose(self):
        self.assertEqual(parse_json_response('{"value": 1}'), {"value": 1})
        self.assertEqual(parse_json_response('```json\n{"value": 1}\n```'), {"value": 1})
        with self.assertRaises(SummaryValidationError):
            parse_json_response('Result: {"value": 1}')
        with self.assertRaises(SummaryValidationError):
            parse_json_response('{"value": NaN}')

    def test_validates_provenance_against_real_and_allowed_blocks(self):
        summary = {
            "invoice_number": "FAC-42",
            "provenance": [
                {
                    "field_path": "/invoice_number",
                    "page_number": 1,
                    "source_block_ids": ["p1_b0"],
                    "raw_text": "Facture FAC-42",
                    "ocr_confidence": 0.9,
                }
            ],
        }
        document = {
            "pages": [
                {
                    "page_number": 1,
                    "blocks": [
                        {"block_id": "p1_b0", "text": "Facture FAC-42"}
                    ],
                }
            ]
        }

        validate_provenance(summary, document, allowed_block_ids={"p1_b0"})
        with self.assertRaises(SummaryValidationError):
            validate_provenance(summary, document, allowed_block_ids=set())

    def test_detects_accounting_inconsistencies(self):
        summary = {
            "amounts": {
                "subtotal_excl_tax": 1000.0,
                "shipping_amount": 0.0,
                "other_charges": 0.0,
                "tax_amount": 200.0,
                "stamp_duty": 0.0,
                "total_incl_tax": 1300.0,
                "deposit_amount": 100.0,
                "amount_paid": 0.0,
                "amount_due": 1200.0,
            }
        }

        warnings = accounting_warnings(summary)

        self.assertEqual(len(warnings), 1)
        self.assertIn("total TTC", warnings[0])


if __name__ == "__main__":
    unittest.main()
