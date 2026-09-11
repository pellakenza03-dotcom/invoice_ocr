from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from invoice_extract.cli import main
from invoice_extract.service import InvoiceExtractionService


OCR_TEXT = (
    "FACTURE FAC-42 11/09/2026 EUR Supplier SAS Customer SARL "
    "Total TTC 1200,00 Net à payer 1200,00"
)


def canonical_document() -> dict:
    return {
        "schema_version": "1.0.0",
        "document_id": "invoice-42",
        "source_file": "invoice.png",
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "width": 1000,
                "height": 1000,
                "orientation": 0,
                "blocks": [
                    {
                        "block_id": "p1_b0",
                        "type": "text",
                        "text": OCR_TEXT,
                        "bbox": [10, 10, 900, 100],
                        "ocr_confidence": 0.95,
                        "reading_order": 0,
                    }
                ],
                "tables": [],
            }
        ],
    }


def model_summary() -> dict:
    empty_party = {
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
    evidence_values = {
        "/invoice_number": "FAC-42",
        "/issue_date": "11/09/2026",
        "/currency": "EUR",
        "/seller/name": "Supplier SAS",
        "/buyer/name": "Customer SARL",
        "/amounts/total_incl_tax": "Total TTC 1200,00",
        "/amounts/amount_due": "Net à payer 1200,00",
    }
    return {
        "schema_version": "1.0.0",
        "document_id": "model-must-not-control-this",
        "document_type": "invoice",
        "invoice_number": " FAC-42 ",
        "issue_date": "11/09/2026",
        "due_date": None,
        "currency": "EUR",
        "seller": {**empty_party, "name": "Supplier SAS"},
        "buyer": {**empty_party, "name": "Customer SARL"},
        "amounts": {
            "subtotal_excl_tax": None,
            "discount_amount": None,
            "shipping_amount": None,
            "other_charges": None,
            "tax_amount": None,
            "stamp_duty": None,
            "total_incl_tax": "1 200,00",
            "deposit_amount": None,
            "amount_paid": None,
            "amount_due": "1 200,00",
        },
        "vat_breakdown": [],
        "provenance": [
            {
                "field_path": path,
                "page_number": 1,
                "source_block_ids": ["p1_b0"],
                "raw_text": raw_text,
                "ocr_confidence": 0.95,
            }
            for path, raw_text in evidence_values.items()
        ],
        "extraction_status": "complete",
        "warnings": [],
        "extraction_metadata": {
            "provider": "model-output",
            "model": "model-output",
            "prompt_version": "model-output",
            "generated_at": "2026-01-01T00:00:00Z",
            "duration_ms": None,
        },
    }


class FakeLLM:
    provider = "test"
    model_id = "fake-qwen"

    def __init__(self, response: dict):
        self.response = response
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return json.dumps(self.response, ensure_ascii=False)


class InvoiceExtractTests(unittest.TestCase):
    def test_extracts_normalizes_validates_and_overrides_metadata(self):
        backend = FakeLLM(model_summary())
        service = InvoiceExtractionService(backend=backend)

        result = service.extract_document(canonical_document())

        self.assertEqual(result["document_id"], "invoice-42")
        self.assertEqual(result["issue_date"], "2026-09-11")
        self.assertEqual(result["amounts"]["amount_due"], 1200.0)
        self.assertEqual(result["extraction_status"], "complete")
        self.assertEqual(result["extraction_metadata"]["provider"], "test")
        self.assertIn("<output_schema>", backend.messages[1]["content"])
        self.assertIn("p1_b0", backend.messages[1]["content"])

    def test_context_only_cli_never_requires_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "invoice-42.ocr.json"
            output = root / "contexts"
            source.write_text(json.dumps(canonical_document()), encoding="utf-8")

            exit_code = main(
                [str(source), "--output", str(output), "--context-only"]
            )

            self.assertEqual(exit_code, 0)
            context = (output / "invoice-42.context.txt").read_text(encoding="utf-8")
            self.assertIn("p1_b0", context)
            self.assertIn("FACTURE FAC-42", context)


if __name__ == "__main__":
    unittest.main()
