"""Command-line entry point for the project's persistent OCR integration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from .pipeline import (
    DEFAULT_CONFIG_PATH,
    OCRConfigurationError,
    OCRInputError,
    PaddleOCRService,
    get_ocr_service,
)

DEFAULT_OUTPUT_DIR = Path("data/ocr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ocr",
        description="Run PP-StructureV3 and save its native JSON results.",
    )
    parser.add_argument("input", type=Path, help="PDF or image to process")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="OCR JSON configuration file",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[str | Path], PaddleOCRService] = get_ocr_service,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        service = service_factory(args.config)
        results = service.predict(args.input)
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for page_number, result in enumerate(results, start=1):
            if not hasattr(result, "save_to_json"):
                raise RuntimeError(
                    "PaddleOCR returned a result that cannot be saved as JSON."
                )
            result.save_to_json(str(DEFAULT_OUTPUT_DIR))
            print(f"Page {page_number}/{len(results)} saved")

        print(f"OCR completed: {len(results)} page(s) -> {DEFAULT_OUTPUT_DIR.resolve()}")
        return 0
    except (OCRConfigurationError, OCRInputError, RuntimeError) as exc:
        print(f"OCR error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
