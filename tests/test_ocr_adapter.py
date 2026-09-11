from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocr_adapter.adapter import (
    AdapterError,
    adapt_document,
    adapt_page,
    discover_document_groups,
    validate_canonical_document,
)
from ocr_adapter.cli import main


def native_page(*, width=1000, height=1000):
    return {
        "input_path": "/missing/invoice.png",
        "width": width,
        "height": height,
        "doc_preprocessor_res": {"angle": -1},
        "layout_det_res": {
            "boxes": [
                {
                    "label": "header",
                    "score": 0.9,
                    "coordinate": [10, 10, 500, 200],
                },
                {
                    "label": "table",
                    "score": 0.8,
                    "coordinate": [10, 300, 900, 700],
                },
            ]
        },
        "overall_ocr_res": {
            "rec_texts": ["Supplier", "Description", "42.00"],
            "rec_scores": [0.95, 0.9, 0.85],
            "rec_boxes": [
                [20, 20, 200, 60],
                [20, 320, 250, 360],
                [700, 320, 850, 360],
            ],
            "rec_polys": [
                [[20, 20], [200, 20], [200, 60], [20, 60]],
                [[20, 320], [250, 320], [250, 360], [20, 360]],
                [[700, 320], [850, 320], [850, 360], [700, 360]],
            ],
        },
        "table_res_list": [
            {
                "pred_html": (
                    "<table><tr><th>Description</th><th>Total</th></tr>"
                    "<tr><td>Service</td><td>42.00</td></tr></table>"
                ),
                "cell_box_list": [
                    [20, 320, 250, 360],
                    [700, 320, 850, 360],
                    [20, 400, 250, 440],
                    [700, 400, 850, 440],
                ],
            }
        ],
    }


class OCRAdapterTests(unittest.TestCase):
    def test_preserves_every_ocr_line_and_builds_table_cells(self):
        page = adapt_page(native_page(), 1)

        self.assertEqual([block["text"] for block in page["blocks"]], [
            "Supplier", "Description", "42.00"
        ])
        self.assertEqual([block["type"] for block in page["blocks"]], [
            "header", "table", "table"
        ])
        self.assertEqual(len(page["tables"]), 1)
        self.assertEqual(len(page["tables"][0]["cells"]), 4)
        self.assertEqual(page["tables"][0]["cells"][1]["column"], 1)

    def test_ignores_suspicious_full_page_table_but_keeps_ocr(self):
        native = native_page()
        native["layout_det_res"]["boxes"] = [
            {
                "label": "table",
                "score": 0.99,
                "coordinate": [0, 0, 1000, 900],
            }
        ]

        page = adapt_page(native, 1)

        self.assertEqual(len(page["blocks"]), 3)
        self.assertTrue(all(block["type"] == "text" for block in page["blocks"]))
        self.assertEqual(page["tables"], [])
        self.assertTrue(any("suspicious table" in warning for warning in page["warnings"]))

    def test_table_cell_ignores_overlapping_visual_block_without_ocr_confidence(self):
        native = native_page()
        native["layout_det_res"]["boxes"].append(
            {
                "label": "image",
                "score": 0.92,
                "coordinate": [20, 400, 250, 440],
            }
        )

        page = adapt_page(native, 1)

        visual_blocks = [block for block in page["blocks"] if block["type"] == "image"]
        self.assertEqual(len(visual_blocks), 1)
        cell = page["tables"][0]["cells"][2]
        self.assertEqual(cell["ocr_confidence"], 0.0)
        self.assertNotIn("source_block_ids", cell)

    def test_groups_and_validates_multipage_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for page_number in (2, 1):
                path = root / f"invoice__p{page_number:03d}.structure.json"
                path.write_text(json.dumps(native_page()), encoding="utf-8")

            groups = discover_document_groups(root)
            document = adapt_document(
                groups["invoice"], generated_at="2026-09-10T12:00:00Z"
            )
            validate_canonical_document(document)

        self.assertEqual(document["document_id"], "invoice")
        self.assertEqual(document["page_count"], 2)
        self.assertEqual([page["page_number"] for page in document["pages"]], [1, 2])

    def test_rejects_non_contiguous_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invoice__p002.structure.json"
            path.write_text(json.dumps(native_page()), encoding="utf-8")
            with self.assertRaises(AdapterError):
                adapt_document([(2, path)])

    def test_cli_writes_canonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "invoice.structure.json"
            output = root / "canonical"
            source.write_text(json.dumps(native_page()), encoding="utf-8")

            exit_code = main([str(source), "--output", str(output)])

            self.assertEqual(exit_code, 0)
            result = json.loads((output / "invoice.ocr.json").read_text())
            self.assertEqual(result["page_count"], 1)


if __name__ == "__main__":
    unittest.main()
