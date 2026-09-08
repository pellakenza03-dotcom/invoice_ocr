"""Compatibility CLI exporting layout from the shared PP-StructureV3 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from ocr.pipeline import (
    DEFAULT_CONFIG_PATH,
    OCRConfig,
    OCRConfigurationError,
    OCRInputError,
    PaddleOCRService,
)

from .coco import COCOValidationError, build_coco_preannotations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "config" / "layout_audit_profiles.json"
LABELS_PATH = PROJECT_ROOT / "config" / "layout_labels.json"
DEFAULT_OUTPUT_ROOT = Path("data/layout_training/preannotations")
PROFILE_NAMES = ("default", "low_threshold")
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m layout_detect",
        description=(
            "Run the shared PP-StructureV3 pipeline and export its layout "
            "result, or build validated COCO pre-annotations."
        ),
    )
    parser.add_argument("input", type=Path, help="Training image or image directory")
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="default",
        help="Layout post-processing profile (default: default)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Shared OCR JSON configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory (default: "
            "data/layout_training/preannotations/<profile>)"
        ),
    )
    parser.add_argument(
        "--build-coco",
        action="store_true",
        help="Build COCO from existing native JSON without loading the model",
    )
    parser.add_argument(
        "--labels-config",
        type=Path,
        default=LABELS_PATH,
        help="Target layout labels and source-label mapping JSON",
    )
    parser.add_argument(
        "--coco-output",
        type=Path,
        help=(
            "COCO output file (default: "
            "<dataset>/annotations/preannotations.coco.json)"
        ),
    )
    return parser


def discover_images(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise OCRInputError(
                "Layout pre-annotation accepts prepared page images only, not PDFs. "
                "Run prepare_layout_data first."
            )
        return [path]
    if not path.is_dir():
        raise OCRInputError(f"Layout input does not exist: {path}")
    images = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda item: item.name.casefold(),
    )
    if not images:
        raise OCRInputError(f"No supported training image found in: {path}")
    return images


def load_profile(name: str) -> dict[str, object]:
    try:
        profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OCRConfigurationError(
            f"Layout profiles file not found: {PROFILES_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OCRConfigurationError(f"Invalid layout profiles JSON: {exc}") from exc
    profile = profiles.get(name) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise OCRConfigurationError(f"Layout profile not found: {name}")
    return profile


def save_result(result: object, image: Path, output_dir: Path) -> int:
    try:
        layout_result = result["layout_det_res"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("PP-StructureV3 returned no layout result.") from exc
    if not hasattr(layout_result, "save_to_json") or not hasattr(
        layout_result, "save_to_img"
    ):
        raise RuntimeError("PP-StructureV3 returned a non-exportable layout result.")
    json_path = output_dir / f"{image.stem}.layout.json"
    visualization_path = output_dir / f"{image.stem}.layout.png"
    layout_result.save_to_json(str(json_path))
    layout_result.save_to_img(str(visualization_path))
    try:
        return len(layout_result["boxes"])
    except (KeyError, TypeError):
        return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[OCRConfig], PaddleOCRService] = PaddleOCRService,
    output_dir_override: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        images = discover_images(args.input)
        output_dir = (
            output_dir_override
            or args.output
            or DEFAULT_OUTPUT_ROOT / args.profile
        )

        if args.build_coco:
            input_path = args.input.expanduser().resolve()
            if not input_path.is_dir():
                raise COCOValidationError(
                    "--build-coco requires the complete training images directory."
                )
            dataset_root = input_path.parent
            coco_output = (
                args.coco_output
                or dataset_root / "annotations" / "preannotations.coco.json"
            )
            summary = build_coco_preannotations(
                images_dir=input_path,
                manifest_path=dataset_root / "manifest.csv",
                preannotations_dir=output_dir,
                labels_path=args.labels_config,
                output_path=coco_output,
            )
            print("COCO pre-annotations created")
            print(f"Images: {summary['image_count']}")
            print(f"Annotations: {summary['annotation_count']}")
            print(f"Categories: {summary['category_count']}")
            for name, count in summary["class_counts"].items():
                print(f"  {name}: {count}")
            print(f"Output: {summary['output_path']}")
            return 0

        config = OCRConfig.from_json(args.config).with_overrides(
            load_profile(args.profile)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Loading PP-StructureV3 once on {config.device} with "
            f"{config.layout_detection_model_name} "
            f"(threshold={config.layout_threshold})..."
        )
        service = service_factory(config)

        failures = 0
        detection_count = 0
        for index, image in enumerate(images, start=1):
            print(f"[{index}/{len(images)}] {image.name}")
            try:
                results = service.predict(image)
                if len(results) != 1:
                    raise RuntimeError(
                        f"Expected one result for one image, received {len(results)}."
                    )
                count = save_result(results[0], image, output_dir)
                detection_count += count
                print(f"  {count} region(s) saved")
            except Exception as exc:
                failures += 1
                print(f"  Failed: {exc}", file=sys.stderr)

        print(
            f"Layout detection completed: {len(images) - failures}/{len(images)} "
            f"image(s), {detection_count} region(s) -> {output_dir.resolve()}"
        )
        return 1 if failures else 0
    except (
        COCOValidationError,
        OCRConfigurationError,
        OCRInputError,
        RuntimeError,
    ) as exc:
        print(f"Layout detection error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
