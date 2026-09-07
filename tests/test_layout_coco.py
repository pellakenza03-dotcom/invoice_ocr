from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from layout_detect.coco import COCOValidationError, build_coco_preannotations
from layout_detect.cli import main


CATEGORIES = [
    {"id": 1, "name": "doc_title"},
    {"id": 2, "name": "header"},
    {"id": 3, "name": "footer"},
    {"id": 4, "name": "text"},
    {"id": 5, "name": "table"},
    {"id": 6, "name": "image"},
]


def create_fixture(root: Path) -> dict[str, Path]:
    dataset = root / "layout_training"
    images = dataset / "images"
    preannotations = dataset / "preannotations" / "low_threshold"
    images.mkdir(parents=True)
    preannotations.mkdir(parents=True)

    Image.new("RGB", (100, 80), "white").save(images / "a.png")
    Image.new("RGB", (120, 90), "white").save(images / "b.png")
    manifest = dataset / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "source_document", "page", "width", "height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image": "images/a.png",
                "source_document": "a.pdf",
                "page": 1,
                "width": 100,
                "height": 80,
            }
        )
        writer.writerow(
            {
                "image": "images/b.png",
                "source_document": "b.pdf",
                "page": 1,
                "width": 120,
                "height": 90,
            }
        )

    (preannotations / "a.layout.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "label": "paragraph_title",
                        "score": 0.8,
                        "coordinate": [10, 15, 60, 55],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (preannotations / "b.layout.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "label": "table",
                        "score": 0.9,
                        "coordinate": [5, 10, 105, 80],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    labels = root / "layout_labels.json"
    labels.write_text(
        json.dumps(
            {
                "categories": CATEGORIES,
                "source_label_mapping": {
                    "paragraph_title": "text",
                    "table": "table",
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "dataset": dataset,
        "images": images,
        "manifest": manifest,
        "preannotations": preannotations,
        "labels": labels,
        "output": dataset / "annotations" / "preannotations.coco.json",
    }


def build(paths: dict[str, Path]):
    return build_coco_preannotations(
        images_dir=paths["images"],
        manifest_path=paths["manifest"],
        preannotations_dir=paths["preannotations"],
        labels_path=paths["labels"],
        output_path=paths["output"],
    )


class LayoutCOCOTests(unittest.TestCase):
    def test_builds_deterministic_valid_coco_and_maps_source_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = create_fixture(Path(directory))

            summary = build(paths)
            coco = json.loads(paths["output"].read_text(encoding="utf-8"))

            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["annotation_count"], 2)
            self.assertEqual([image["id"] for image in coco["images"]], [1, 2])
            self.assertEqual(
                [annotation["id"] for annotation in coco["annotations"]], [1, 2]
            )
            self.assertEqual(coco["annotations"][0]["category_id"], 4)
            self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 15.0, 50.0, 40.0])
            self.assertEqual(len(coco["categories"]), 6)

    def test_cli_build_coco_does_not_load_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = create_fixture(Path(directory))

            exit_code = main(
                [
                    str(paths["images"]),
                    "--profile",
                    "low_threshold",
                    "--build-coco",
                    "--output",
                    str(paths["preannotations"]),
                    "--labels-config",
                    str(paths["labels"]),
                    "--coco-output",
                    str(paths["output"]),
                ],
                service_factory=lambda _config: self.fail("model must not load"),
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(paths["output"].is_file())

    def test_rejects_missing_image_and_missing_preannotation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = create_fixture(Path(directory))
            (paths["images"] / "a.png").unlink()
            with self.assertRaisesRegex(COCOValidationError, "does not exist"):
                build(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = create_fixture(Path(directory))
            (paths["preannotations"] / "a.layout.json").unlink()
            with self.assertRaisesRegex(COCOValidationError, "does not exist"):
                build(paths)

    def test_rejects_dimension_mismatch_and_invalid_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = create_fixture(Path(directory))
            manifest_text = paths["manifest"].read_text(encoding="utf-8")
            paths["manifest"].write_text(
                manifest_text.replace("a.pdf,1,100,80", "a.pdf,1,101,80"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(COCOValidationError, "dimensions"):
                build(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = create_fixture(Path(directory))
            labels = json.loads(paths["labels"].read_text(encoding="utf-8"))
            labels["categories"].pop()
            paths["labels"].write_text(json.dumps(labels), encoding="utf-8")
            with self.assertRaisesRegex(COCOValidationError, "Exactly 6"):
                build(paths)

    def test_rejects_unmapped_nonnumeric_and_out_of_bounds_boxes(self):
        invalid_cases = (
            ("unknown", [1, 2, 10, 20], "Unmapped"),
            ("table", [1, "bad", 10, 20], "Non-numeric"),
            ("table", [1, 2, 130, 20], "Out-of-bounds"),
            ("table", [20, 2, 10, 20], "non-positive"),
        )
        for label, coordinate, error in invalid_cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                paths = create_fixture(Path(directory))
                result_path = paths["preannotations"] / "a.layout.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["boxes"][0]["label"] = label
                result["boxes"][0]["coordinate"] = coordinate
                result_path.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaisesRegex(COCOValidationError, error):
                    build(paths)


if __name__ == "__main__":
    unittest.main()
