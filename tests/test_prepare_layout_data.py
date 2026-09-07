import csv
import tempfile
import unittest
from pathlib import Path

import pymupdf
from PIL import Image

from prepare_layout_data.cli import (
    PreparationError,
    prepare_dataset,
    refresh_manifest,
)


class PrepareLayoutDataTests(unittest.TestCase):
    def test_renders_pdf_pages_copies_images_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            Image.new("RGB", (80, 40), "white").save(input_dir / "Facture été.jpg")
            pdf = pymupdf.open()
            pdf.new_page(width=144, height=72)
            pdf.new_page(width=72, height=144)
            pdf.save(input_dir / "Deux pages.pdf")
            pdf.close()

            rows = prepare_dataset(input_dir, output_dir, dpi=100)

            self.assertEqual(len(rows), 3)
            self.assertEqual(
                sorted(path.name for path in (output_dir / "images").iterdir()),
                ["deux-pages__p001.png", "deux-pages__p002.png", "facture-ete.jpg"],
            )
            self.assertEqual(rows[0]["source_type"], "pdf")
            self.assertEqual(rows[0]["width"], 200)
            self.assertEqual(rows[0]["height"], 100)
            self.assertEqual(rows[2]["width"], 80)
            self.assertEqual(rows[2]["height"], 40)

            with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as file:
                manifest_rows = list(csv.DictReader(file))
            self.assertEqual(len(manifest_rows), 3)
            self.assertEqual(manifest_rows[1]["page"], "2")
            self.assertEqual(len(manifest_rows[0]["sha256"]), 64)
            self.assertEqual(manifest_rows[0]["include_in_training"], "yes")
            self.assertIn(manifest_rows[0]["audit_status"], {"keep", "review"})
            self.assertEqual(
                manifest_rows[2]["source_document"], rows[2]["source_document"]
            )

    def test_flags_exact_duplicates_without_excluding_them(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            image = Image.new("RGB", (20, 20), "white")
            image.save(input_dir / "a.png")
            image.save(input_dir / "b.png")

            rows = prepare_dataset(input_dir, output_dir)

            self.assertEqual(rows[0]["duplicate_group"], "dup_001")
            self.assertEqual(rows[1]["duplicate_group"], "dup_001")
            self.assertIn("exact_duplicate", rows[0]["audit_flags"])
            self.assertEqual(rows[0]["audit_status"], "review")
            self.assertEqual(rows[0]["include_in_training"], "yes")

    def test_refreshes_existing_manifest_without_recopying_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            Image.new("RGB", (10, 10), "black").save(input_dir / "invoice.png")
            prepare_dataset(input_dir, output_dir)
            image_path = output_dir / "images" / "invoice.png"
            original_mtime = image_path.stat().st_mtime_ns
            manifest_path = output_dir / "manifest.csv"
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(
                manifest_text.replace(",yes\n", ",no\n"), encoding="utf-8"
            )

            rows = refresh_manifest(output_dir)

            self.assertEqual(len(rows), 1)
            self.assertEqual(image_path.stat().st_mtime_ns, original_mtime)
            self.assertEqual(rows[0]["audit_status"], "keep")
            self.assertEqual(rows[0]["include_in_training"], "no")

    def test_refuses_to_overwrite_an_existing_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            Image.new("RGB", (10, 10), "white").save(input_dir / "invoice.png")

            prepare_dataset(input_dir, output_dir)

            with self.assertRaisesRegex(PreparationError, "not empty"):
                prepare_dataset(input_dir, output_dir)

    def test_rejects_non_positive_dpi(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(PreparationError, "positive"):
                prepare_dataset(root, root / "output", dpi=0)


if __name__ == "__main__":
    unittest.main()
