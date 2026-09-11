from __future__ import annotations

import unittest

from invoice_extract.context import build_ocr_context


def ocr_document(block_count: int = 3) -> dict:
    blocks = [
        {
            "block_id": f"p1_b{index}",
            "type": "text",
            "text": f"Text line {index} with useful invoice content",
            "bbox": [10, 20 + index * 30, 500, 40 + index * 30],
            "ocr_confidence": 0.9,
            "reading_order": index,
        }
        for index in range(block_count)
    ]
    blocks.append(
        {
            "block_id": f"p1_b{block_count}",
            "type": "image",
            "bbox": [600, 100, 900, 300],
            "reading_order": block_count,
        }
    )
    return {
        "schema_version": "1.0.0",
        "document_id": "invoice-001",
        "source_file": "invoice.png",
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "width": 1000,
                "height": 1000,
                "orientation": 0,
                "blocks": blocks,
                "tables": [],
            }
        ],
    }


class InvoiceContextTests(unittest.TestCase):
    def test_renders_text_blocks_with_normalized_geometry(self):
        context = build_ocr_context(ocr_document(), coordinate_scale=1000)

        self.assertIn("DOCUMENT_ID: \"invoice-001\"", context.text)
        self.assertIn("bbox=10,20,500,40", context.text)
        self.assertIn('"Text line 0 with useful invoice content"', context.text)
        self.assertNotIn("type=image", context.text)
        self.assertEqual(context.included_block_ids, ("p1_b0", "p1_b1", "p1_b2"))
        self.assertFalse(context.truncated)

    def test_truncation_preserves_first_and_last_ocr_blocks(self):
        context = build_ocr_context(ocr_document(30), max_characters=700)

        self.assertTrue(context.truncated)
        self.assertIn("p1_b0", context.included_block_ids)
        self.assertIn("p1_b29", context.included_block_ids)
        self.assertIn("OCR BLOCKS OMITTED", context.text)
        self.assertLessEqual(len(context.text), 700)


if __name__ == "__main__":
    unittest.main()
