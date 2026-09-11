"""Command-line interface for semantic invoice-summary extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .context import ContextError, load_ocr_document
from .service import DEFAULT_CONFIG_PATH, ExtractionError, InvoiceExtractionService


DEFAULT_OUTPUT_DIR = Path("data/invoice_results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m invoice_extract",
        description="Extract seller, buyer and invoice totals from canonical OCR JSON.",
    )
    parser.add_argument("input", type=Path, help="Canonical *.ocr.json file or directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/invoice_results)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Extraction configuration JSON",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Write LLM context files without loading Qwen",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing non-empty output files",
    )
    return parser


def _discover(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if not resolved.name.lower().endswith(".ocr.json"):
            raise ExtractionError(f"Expected a *.ocr.json file: {resolved}")
        return [resolved]
    if resolved.is_dir():
        files = sorted(resolved.glob("*.ocr.json"), key=lambda item: item.name.casefold())
        if files:
            return files
        raise ExtractionError(f"No *.ocr.json files found in: {resolved}")
    raise ExtractionError(f"Extraction input does not exist: {resolved}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = _discover(args.input)
        service = InvoiceExtractionService(args.config)
        args.output.mkdir(parents=True, exist_ok=True)
        completed = 0
        for index, source in enumerate(files, start=1):
            document = load_ocr_document(source, service.config["ocr_schema_path"])
            document_id = document["document_id"]
            suffix = ".context.txt" if args.context_only else ".invoice.json"
            destination = args.output / f"{document_id}{suffix}"
            if args.resume and destination.is_file() and destination.stat().st_size > 0:
                print(f"[{index}/{len(files)}] Skipped: {document_id}")
                completed += 1
                continue
            if args.context_only:
                context = service.prepare_context(document)
                destination.write_text(context.text + "\n", encoding="utf-8")
                detail = (
                    f"{len(context.included_block_ids)} OCR block(s), "
                    f"{len(context.omitted_block_ids)} omitted"
                )
            else:
                summary = service.extract_document(document)
                destination.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                detail = summary["extraction_status"]
            completed += 1
            print(f"[{index}/{len(files)}] {document_id}: {detail} -> {destination}")
        print(f"Invoice extraction completed: {completed}/{len(files)} document(s)")
        return 0
    except (ContextError, ExtractionError) as exc:
        print(f"Invoice extraction error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
