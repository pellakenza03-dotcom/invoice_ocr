"""CLI for converting native PP-StructureV3 exports to canonical OCR JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .adapter import (
    DEFAULT_SCHEMA_PATH,
    AdapterError,
    adapt_document,
    discover_document_groups,
    validate_canonical_document,
)


DEFAULT_OUTPUT_DIR = Path("data/ocr_canonical")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_adapter",
        description="Convert PP-StructureV3 *.structure.json files to canonical OCR JSON.",
    )
    parser.add_argument("input", type=Path, help="Structure JSON file or directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Canonical output directory (default: data/ocr_canonical)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Canonical OCR JSON Schema",
    )
    parser.add_argument(
        "--max-table-page-ratio",
        type=float,
        default=0.75,
        help="Ignore table structure above this fraction of the page (default: 0.75)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 0 < args.max_table_page_ratio <= 1:
            raise AdapterError("--max-table-page-ratio must be in the interval (0, 1].")
        groups = discover_document_groups(args.input)
        args.output.mkdir(parents=True, exist_ok=True)
        for index, (document_id, pages) in enumerate(groups.items(), start=1):
            document = adapt_document(
                pages,
                document_id=document_id,
                max_table_page_ratio=args.max_table_page_ratio,
            )
            validate_canonical_document(document, args.schema)
            destination = args.output / f"{document_id}.ocr.json"
            destination.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            block_count = sum(len(page["blocks"]) for page in document["pages"])
            table_count = sum(len(page["tables"]) for page in document["pages"])
            warning_count = len(document.get("warnings", []))
            print(
                f"[{index}/{len(groups)}] {document_id}: "
                f"{len(pages)} page(s), {block_count} block(s), "
                f"{table_count} table(s), {warning_count} warning(s) -> {destination}"
            )
        print(f"Canonical OCR completed: {len(groups)} document(s) -> {args.output.resolve()}")
        return 0
    except AdapterError as exc:
        print(f"OCR adapter error: {exc}", file=sys.stderr)
        return 1
