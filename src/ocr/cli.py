"""Command-line entry point for the project's persistent OCR integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from .pipeline import (
    DEFAULT_CONFIG_PATH,
    SUPPORTED_INPUT_SUFFIXES,
    OCRConfig,
    OCRConfigurationError,
    OCRInputError,
    PaddleOCRService,
)

DEFAULT_OUTPUT_DIR = Path("data/ocr")
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
        "--audit-profile",
        choices=AUDIT_PROFILE_NAMES,
        help="Apply an audit profile and save under data/layout_audit/output",
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

    for page_number, result in enumerate(results, start=1):
        if not hasattr(result, "save_to_json"):
            raise RuntimeError(
                "PaddleOCR returned a result that cannot be saved as JSON."
            )
        result.save_to_json(str(output_dir))

        layout_result = result.get("layout_det_res")
        if layout_result is None or not hasattr(layout_result, "save_to_img"):
            raise RuntimeError(
                "PaddleOCR returned no exportable layout detection result."
            )
        layout_path = output_dir / (
            f"{document.stem}_{page_number - 1}_layout_det_res.png"
        )
        layout_result.save_to_img(str(layout_path))
        print(f"  Page {page_number}/{len(results)} saved")
    return len(results)


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[OCRConfig], PaddleOCRService] = PaddleOCRService,
    output_dir_override: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = OCRConfig.from_json(args.config)
        if args.audit_profile:
            config = config.with_overrides(load_audit_overrides(args.audit_profile))

        output_dir = output_dir_override or (
            AUDIT_OUTPUT_ROOT / args.audit_profile
            if args.audit_profile
            else DEFAULT_OUTPUT_DIR
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        documents = discover_documents(args.input)
        service = service_factory(config)

        page_count = 0
        failures = 0
        for index, document in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] {document.name}")
            try:
                results = service.predict(document)
                page_count += save_page_results(results, document, output_dir)
            except Exception as exc:
                failures += 1
                print(f"  Failed: {exc}", file=sys.stderr)

        print(
            f"OCR completed: {len(documents) - failures}/{len(documents)} document(s), "
            f"{page_count} page(s) -> {output_dir.resolve()}"
        )
        return 1 if failures else 0
    except (OCRConfigurationError, OCRInputError, RuntimeError) as exc:
        print(f"OCR error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
