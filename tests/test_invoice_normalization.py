from __future__ import annotations

import unittest

from invoice_extract.normalize import (
    NormalizationError,
    normalize_currency,
    normalize_date,
    normalize_number,
    normalize_summary,
)


class InvoiceNormalizationTests(unittest.TestCase):
    def test_normalizes_european_and_international_amounts(self):
        cases = {
            "1 200,50 €": 1200.5,
            "1.200,50": 1200.5,
            "1,200.50": 1200.5,
            "(300,00)": -300.0,
            "1 200": 1200.0,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_number(raw), expected)

    def test_normalizes_dates_and_currencies(self):
        self.assertEqual(normalize_date("11/09/2026"), "2026-09-11")
        self.assertEqual(normalize_currency("€"), "EUR")
        self.assertEqual(normalize_currency("dh"), "MAD")

    def test_rejects_ambiguous_non_numeric_values(self):
        with self.assertRaises(NormalizationError):
            normalize_number("unknown")
        with self.assertRaises(NormalizationError):
            normalize_date("September someday")
        with self.assertRaises(NormalizationError):
            normalize_number(float("nan"))

    def test_normalizes_known_summary_fields_without_creating_fields(self):
        summary = {
            "invoice_number": "  FAC   42 ",
            "issue_date": "11-09-2026",
            "currency": "euros",
            "seller": {"name": "  Supplier  SAS ", "email": "INFO@EXAMPLE.COM"},
            "amounts": {"amount_due": "1 200,50 €"},
            "vat_breakdown": [{"rate": "20%", "tax_amount": "200,00"}],
        }

        result = normalize_summary(summary)

        self.assertEqual(result["invoice_number"], "FAC 42")
        self.assertEqual(result["currency"], "EUR")
        self.assertEqual(result["seller"]["email"], "info@example.com")
        self.assertEqual(result["amounts"]["amount_due"], 1200.5)
        self.assertNotIn("due_date", result)


if __name__ == "__main__":
    unittest.main()
