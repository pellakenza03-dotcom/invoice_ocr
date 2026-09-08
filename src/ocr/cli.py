"""Command-line entry point for the project's persistent OCR integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from layout_detect.coco import COCOValidationError, build_coco_preannotations

from .pipeline import (
    DEFAULT_CONFIG_PATH,
    SUPPORTED_INPUT_SUFFIXES,
    OCRConfig,
    OCRConfigurationError,
    OCRInputError,
    PaddleOCRService,
)

DEFAULT_OUTPUT_DIR = Path("data/ocr")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "config" / "layout_labels.json"
AUDIT_OUTPUT_ROOT = Path("data/layout_audit/output")
AUDIT_PROFILES_PATH = DEFAULT_CONFIG_PATH.with_name("layout_audit_profiles.json")
AUDIT_PROFILE_NAMES = ("default", "low_threshold")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ocr",
        description=(
            "Run PP-StructureV3 and save its native JSON and layout detection image."
        ),
    )
    parser.add_argument(
        "input", type=Path, help="PDF/image file or directory to process"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="OCR JSON configuration file",
    )
    parser.add_argument(
        "--device",
        help="Override the configured device (for example: cpu, gpu, gpu:0)",
    )
    parser.add_argument(
        "--audit-profile",
        choices=AUDIT_PROFILE_NAMES,
        help="Apply an audit profile and save under data/layout_audit/output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory for PP-StructureV3 JSON and visualizations",
    )
    parser.add_argument(
        "--build-coco",
        action="store_true",
        help="After inference, convert exported layout results to COCO",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip page images whose structure, layout JSON and layout image exist",
    )
    parser.add_argument(
        "--labels-config",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Target layout labels and source-label mapping JSON",
    )
    parser.add_argument(
        "--coco-output",
        type=Path,
        help="COCO output (default: <dataset>/annotations/preannotations.coco.json)",
    )
    return parser


def discover_documents(input_path: Path) -> list[Path]:
    """Return supported documents in deterministic order."""

    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise OCRInputError(f"Unsupported OCR input type: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise OCRInputError(f"OCR input does not exist: {path}")

    documents = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        ),
        key=lambda item: item.name.casefold(),
    )
    if not documents:
        raise OCRInputError(f"No supported OCR documents found in: {path}")
    return documents


def load_audit_overrides(profile_name: str) -> dict[str, object]:
    try:
        profiles = json.loads(AUDIT_PROFILES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OCRConfigurationError(
            f"Audit profiles file not found: {AUDIT_PROFILES_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OCRConfigurationError(
            f"Invalid JSON in audit profiles file: {exc}"
        ) from exc

    overrides = profiles.get(profile_name) if isinstance(profiles, dict) else None
    if not isinstance(overrides, dict):
        raise OCRConfigurationError(f"Audit profile not found: {profile_name}")
    return overrides


def save_page_results(
    results: list[object], document: Path, output_dir: Path
) -> int:
    """Save native JSON and layout visualization for every document page."""

    is_pdf = document.suffix.lower() == ".pdf"
    if not is_pdf and len(results) != 1:
        raise RuntimeError(
            f"Expected one PP-StructureV3 result for page image {document.name}, "
            f"received {len(results)}."
        )
    for page_number, result in enumerate(results, start=1):
        if not hasattr(result, "save_to_json"):
            raise RuntimeError(
                "PaddleOCR returned a result that cannot be saved as JSON."
            )
        page_stem = (
            f"{document.stem}__p{page_number:03d}" if is_pdf else document.stem
        )
        result.save_to_json(str(output_dir / f"{page_stem}.structure.json"))

        layout_result = result.get("layout_det_res")
        if layout_result is None or not hasattr(layout_result, "save_to_img"):
            raise RuntimeError(
                "PaddleOCR returned no exportable layout detection result."
            )
        if not hasattr(layout_result, "save_to_json"):
            raise RuntimeError(
                "PaddleOCR returned no exportable layout detection JSON."
            )
        layout_result.save_to_json(str(output_dir / f"{page_stem}.layout.json"))
        layout_result.save_to_img(str(output_dir / f"{page_stem}.layout.png"))
        print(f"  Page {page_number}/{len(results)} saved")
    return len(results)


def _page_outputs_are_complete(document: Path, output_dir: Path) -> bool:
    """Return whether a prepared page image already has valid exports."""

    if document.suffix.lower() == ".pdf":
        return False
    structure_path = output_dir / f"{document.stem}.structure.json"
    layout_path = output_dir / f"{document.stem}.layout.json"
    visualization_path = output_dir / f"{document.stem}.layout.png"
    if not visualization_path.is_file() or visualization_path.stat().st_size == 0:
        return False
    try:
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(structure, dict)
        and isinstance(structure.get("layout_det_res"), dict)
        and isinstance(layout, dict)
        and isinstance(layout.get("boxes"), list)
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[OCRConfig], PaddleOCRService] = PaddleOCRService,
    output_dir_override: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = OCRConfig.from_json(args.config)
        if args.device:
            config = config.with_overrides({"device": args.device})
        if args.audit_profile:
            config = config.with_overrides(load_audit_overrides(args.audit_profile))

        output_dir = output_dir_override or args.output or (
            AUDIT_OUTPUT_ROOT / args.audit_profile
            if args.audit_profile
            else DEFAULT_OUTPUT_DIR
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        documents = discover_documents(args.input)
        service: PaddleOCRService | None = None

        page_count = 0
        failures = 0
        for index, document in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] {document.name}")
            if args.resume and _page_outputs_are_complete(document, output_dir):
                page_count += 1
                print("  Already complete; skipped")
                continue
            try:
                if service is None:
                    service = service_factory(config)
                results = service.predict(document)
                page_count += save_page_results(results, document, output_dir)
            except Exception as exc:
                failures += 1
                print(f"  Failed: {exc}", file=sys.stderr)

        print(
            f"OCR completed: {len(documents) - failures}/{len(documents)} document(s), "
            f"{page_count} page(s) -> {output_dir.resolve()}"
        )
        if failures:
            return 1

        if args.build_coco:
            input_path = args.input.expanduser().resolve()
            if not input_path.is_dir():
                raise COCOValidationError(
                    "--build-coco requires the prepared images directory."
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
            print(
                f"COCO pre-annotations created: "
                f"{summary['annotation_count']} region(s) -> "
                f"{summary['output_path']}"
            )
        return 0
    except (
        COCOValidationError,
        OCRConfigurationError,
        OCRInputError,
        RuntimeError,
    ) as exc:
        print(f"OCR error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
