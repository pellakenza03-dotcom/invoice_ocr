"""Create one training image per invoice page and an auditable manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

import pymupdf
from PIL import Image

DEFAULT_INPUT_DIR = Path("data/layout_audit/input")
DEFAULT_OUTPUT_DIR = Path("data/layout_training")
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
MANIFEST_FIELDS = (
    "image",
    "source_document",
    "page",
    "source_type",
    "width",
    "height",
)


class PreparationError(RuntimeError):
    """Raised when the layout dataset cannot be prepared safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prepare_layout_data",
        description=(
            "Render every PDF page as PNG, copy existing images, and create "
            "a manifest.csv for layout annotation."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dataset directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF rendering resolution (default: 200)",
    )
    return parser


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or "document"


def _discover_documents(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise PreparationError(f"Input directory does not exist: {input_dir}")

    documents = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if not documents:
        raise PreparationError(f"No supported PDF or image found in: {input_dir}")
    return documents


def _ensure_empty_destination(output_dir: Path) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() or (images_dir.exists() and any(images_dir.iterdir())):
        raise PreparationError(
            f"Output dataset is not empty: {output_dir}. "
            "Move/delete it explicitly or choose another --output directory."
        )
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, manifest_path


def _unique_stem(document: Path, input_dir: Path, used_stems: set[str]) -> str:
    stem = _slug(document.stem)
    if stem.casefold() not in used_stems:
        used_stems.add(stem.casefold())
        return stem

    relative_name = document.relative_to(input_dir).as_posix()
    digest = hashlib.sha1(relative_name.encode("utf-8")).hexdigest()[:8]
    unique = f"{stem}-{digest}"
    used_stems.add(unique.casefold())
    return unique


def _render_pdf(
    document: Path,
    relative_source: str,
    output_stem: str,
    images_dir: Path,
    dpi: int,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    try:
        with pymupdf.open(document) as pdf:
            if pdf.page_count == 0:
                raise PreparationError(f"PDF has no pages: {document}")
            for page_index, page in enumerate(pdf, start=1):
                filename = f"{output_stem}__p{page_index:03d}.png"
                destination = images_dir / filename
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                pixmap.save(destination)
                rows.append(
                    {
                        "image": f"images/{filename}",
                        "source_document": relative_source,
                        "page": page_index,
                        "source_type": "pdf",
                        "width": pixmap.width,
                        "height": pixmap.height,
                    }
                )
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"Cannot render PDF {document}: {exc}") from exc
    return rows


def _copy_image(
    document: Path,
    relative_source: str,
    output_stem: str,
    images_dir: Path,
) -> dict[str, str | int]:
    suffix = document.suffix.lower()
    filename = f"{output_stem}{suffix}"
    destination = images_dir / filename
    try:
        with Image.open(document) as image:
            width, height = image.size
            image.verify()
        shutil.copy2(document, destination)
    except Exception as exc:
        raise PreparationError(f"Cannot copy image {document}: {exc}") from exc
    return {
        "image": f"images/{filename}",
        "source_document": relative_source,
        "page": 1,
        "source_type": "image",
        "width": width,
        "height": height,
    }


def prepare_dataset(input_dir: Path, output_dir: Path, dpi: int = 200) -> list[dict[str, str | int]]:
    """Prepare page images without modifying any source document."""

    if dpi <= 0:
        raise PreparationError("DPI must be a positive integer.")

    source_root = input_dir.expanduser().resolve()
    destination_root = output_dir.expanduser().resolve()
    documents = _discover_documents(source_root)
    images_dir, manifest_path = _ensure_empty_destination(destination_root)

    rows: list[dict[str, str | int]] = []
    used_stems: set[str] = set()
    for index, document in enumerate(documents, start=1):
        relative_source = document.relative_to(source_root).as_posix()
        output_stem = _unique_stem(document, source_root, used_stems)
        print(f"[{index}/{len(documents)}] {relative_source}")
        if document.suffix.lower() == ".pdf":
            document_rows = _render_pdf(
                document,
                relative_source,
                output_stem,
                images_dir,
                dpi,
            )
            rows.extend(document_rows)
            print(f"  {len(document_rows)} PDF page(s) rendered")
        else:
            rows.append(
                _copy_image(
                    document,
                    relative_source,
                    output_stem,
                    images_dir,
                )
            )
            print("  Image copied")

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = prepare_dataset(args.input, args.output, args.dpi)
        pdf_pages = sum(row["source_type"] == "pdf" for row in rows)
        copied_images = len(rows) - pdf_pages
        print(
            f"Dataset ready: {len(rows)} image(s) "
            f"({pdf_pages} PDF page(s), {copied_images} copied image(s)) "
            f"-> {args.output.resolve()}"
        )
        return 0
    except PreparationError as exc:
        print(f"Preparation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
