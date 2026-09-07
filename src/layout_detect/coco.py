"""Validated conversion of native PaddleX layout results to COCO."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

EXPECTED_CATEGORY_NAMES = (
    "doc_title",
    "header",
    "footer",
    "text",
    "table",
    "image",
)
REQUIRED_MANIFEST_FIELDS = (
    "image",
    "source_document",
    "page",
    "width",
    "height",
)


class COCOValidationError(RuntimeError):
    """Raised when pre-annotations cannot form a valid COCO dataset."""


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise COCOValidationError(f"{description} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise COCOValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _load_labels(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    data = _load_json(path, "Layout labels configuration")
    if not isinstance(data, dict):
        raise COCOValidationError("Layout labels configuration must be an object.")
    categories = data.get("categories")
    mapping = data.get("source_label_mapping")
    if not isinstance(categories, list) or not isinstance(mapping, dict):
        raise COCOValidationError(
            "Layout labels configuration requires categories and source_label_mapping."
        )
    if len(categories) != len(EXPECTED_CATEGORY_NAMES):
        raise COCOValidationError(
            f"Exactly {len(EXPECTED_CATEGORY_NAMES)} COCO categories are required."
        )

    category_ids: list[int] = []
    category_names: list[str] = []
    normalized_categories: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, dict):
            raise COCOValidationError("Each COCO category must be an object.")
        category_id = category.get("id")
        name = category.get("name")
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise COCOValidationError("Every category id must be an integer.")
        if category_id < 1 or not isinstance(name, str) or not name:
            raise COCOValidationError("Every category requires a positive id and name.")
        category_ids.append(category_id)
        category_names.append(name)
        normalized_categories.append(
            {"id": category_id, "name": name, "supercategory": "layout"}
        )

    if len(category_ids) != len(set(category_ids)):
        raise COCOValidationError("COCO category ids must be unique.")
    if len(category_names) != len(set(category_names)):
        raise COCOValidationError("COCO category names must be unique.")
    if set(category_names) != set(EXPECTED_CATEGORY_NAMES):
        raise COCOValidationError(
            "The six required categories are: "
            + ", ".join(EXPECTED_CATEGORY_NAMES)
        )
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items()):
        raise COCOValidationError("Every source label mapping must be string to string.")
    unknown_targets = sorted(set(mapping.values()) - set(category_names))
    if unknown_targets:
        raise COCOValidationError(
            f"Unknown source mapping target(s): {', '.join(unknown_targets)}"
        )
    return normalized_categories, mapping


def _load_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or ())
            missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in fieldnames]
            if missing:
                raise COCOValidationError(
                    f"Manifest is missing field(s): {', '.join(missing)}"
                )
            rows = list(reader)
    except FileNotFoundError as exc:
        raise COCOValidationError(f"Manifest does not exist: {path}") from exc
    if not rows:
        raise COCOValidationError(f"Manifest contains no image: {path}")
    return rows


def _positive_integer(value: str, field: str, image_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise COCOValidationError(
            f"Invalid {field} for {image_name}: {value!r}"
        ) from exc
    if parsed < 1:
        raise COCOValidationError(f"{field} must be positive for {image_name}.")
    return parsed


def _numeric_coordinate(value: object, image_name: str, box_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise COCOValidationError(
            f"Non-numeric coordinate in {image_name}, box {box_index}."
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        raise COCOValidationError(
            f"Non-finite coordinate in {image_name}, box {box_index}."
        )
    return parsed


def _validate_unique_ids(items: list[dict[str, Any]], item_name: str) -> None:
    identifiers = [item.get("id") for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise COCOValidationError(f"{item_name} ids must be unique.")


def build_coco_preannotations(
    *,
    images_dir: Path,
    manifest_path: Path,
    preannotations_dir: Path,
    labels_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate every source and atomically write one COCO pre-annotation file."""

    images_root = images_dir.expanduser().resolve()
    dataset_root = manifest_path.expanduser().resolve().parent
    preannotations_root = preannotations_dir.expanduser().resolve()
    labels_file = labels_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()

    if not images_root.is_dir():
        raise COCOValidationError(f"Images directory does not exist: {images_root}")
    if not preannotations_root.is_dir():
        raise COCOValidationError(
            f"Pre-annotations directory does not exist: {preannotations_root}"
        )

    categories, source_mapping = _load_labels(labels_file)
    category_id_by_name = {category["name"]: category["id"] for category in categories}
    manifest_rows = _load_manifest(manifest_path.expanduser().resolve())

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    manifest_image_paths: list[Path] = []
    seen_relative_images: set[str] = set()

    annotation_id = 1
    for image_id, row in enumerate(manifest_rows, start=1):
        relative_image = row["image"].replace("\\", "/")
        if relative_image in seen_relative_images:
            raise COCOValidationError(
                f"Manifest image path is duplicated: {relative_image}"
            )
        seen_relative_images.add(relative_image)

        image_path = (dataset_root / Path(relative_image)).resolve()
        if not image_path.is_relative_to(dataset_root):
            raise COCOValidationError(
                f"Manifest image escapes the dataset: {relative_image}"
            )
        if image_path.parent != images_root:
            raise COCOValidationError(
                f"Manifest image is outside the selected images directory: {relative_image}"
            )
        if not image_path.is_file():
            raise COCOValidationError(f"Manifest image does not exist: {image_path}")
        manifest_image_paths.append(image_path)

        width = _positive_integer(row["width"], "width", image_path.name)
        height = _positive_integer(row["height"], "height", image_path.name)
        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
                image.verify()
        except Exception as exc:
            raise COCOValidationError(f"Cannot read image {image_path}: {exc}") from exc
        if (width, height) != (actual_width, actual_height):
            raise COCOValidationError(
                f"Manifest dimensions for {image_path.name} are {width}x{height}, "
                f"but the image is {actual_width}x{actual_height}."
            )

        result_path = preannotations_root / f"{image_path.stem}.layout.json"
        result = _load_json(result_path, "Pre-annotation JSON")
        if not isinstance(result, dict) or not isinstance(result.get("boxes"), list):
            raise COCOValidationError(
                f"Pre-annotation boxes must be a list in: {result_path}"
            )

        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for box_index, box in enumerate(result["boxes"], start=1):
            if not isinstance(box, dict):
                raise COCOValidationError(
                    f"Invalid box object in {result_path}, box {box_index}."
                )
            source_label = box.get("label")
            if not isinstance(source_label, str) or source_label not in source_mapping:
                raise COCOValidationError(
                    f"Unmapped source label {source_label!r} in "
                    f"{result_path.name}, box {box_index}."
                )
            target_label = source_mapping[source_label]
            coordinate = box.get("coordinate")
            if not isinstance(coordinate, list) or len(coordinate) != 4:
                raise COCOValidationError(
                    f"Expected four coordinates in {result_path.name}, box {box_index}."
                )
            xmin, ymin, xmax, ymax = [
                _numeric_coordinate(value, image_path.name, box_index)
                for value in coordinate
            ]
            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                raise COCOValidationError(
                    f"Out-of-bounds or non-positive box in {result_path.name}, "
                    f"box {box_index}: {[xmin, ymin, xmax, ymax]} for {width}x{height}."
                )

            score = box.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise COCOValidationError(
                    f"Invalid score in {result_path.name}, box {box_index}."
                )
            score_value = float(score)
            if not math.isfinite(score_value) or not 0 <= score_value <= 1:
                raise COCOValidationError(
                    f"Score must be between 0 and 1 in {result_path.name}, "
                    f"box {box_index}."
                )

            box_width = xmax - xmin
            box_height = ymax - ymin
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id_by_name[target_label],
                    "bbox": [xmin, ymin, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "score": score_value,
                }
            )
            annotation_id += 1
            class_counts[target_label] += 1

    discovered_images = sorted(
        path.resolve()
        for path in images_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    )
    if set(discovered_images) != set(manifest_image_paths):
        missing_from_manifest = sorted(
            path.name for path in set(discovered_images) - set(manifest_image_paths)
        )
        missing_from_directory = sorted(
            path.name for path in set(manifest_image_paths) - set(discovered_images)
        )
        details = []
        if missing_from_manifest:
            details.append("not in manifest: " + ", ".join(missing_from_manifest))
        if missing_from_directory:
            details.append("not in images directory: " + ", ".join(missing_from_directory))
        raise COCOValidationError("Image/manifest mismatch (" + "; ".join(details) + ").")

    _validate_unique_ids(coco_images, "Image")
    _validate_unique_ids(coco_annotations, "Annotation")
    if len(categories) != 6 or set(category_id_by_name) != set(EXPECTED_CATEGORY_NAMES):
        raise COCOValidationError("The six required COCO categories are not present.")

    coco = {
        "info": {
            "description": "Invoice layout pre-annotations",
            "version": "1.0.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "output_path": destination,
        "image_count": len(coco_images),
        "annotation_count": len(coco_annotations),
        "category_count": len(categories),
        "class_counts": {
            name: class_counts.get(name, 0) for name in EXPECTED_CATEGORY_NAMES
        },
    }
