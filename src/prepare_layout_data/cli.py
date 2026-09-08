"""Create one training image per invoice page and an auditable manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

import pymupdf
from PIL import Image, ImageOps

from preprocessing import DocumentOrientationService, normalize_image

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
    "orientation_angle",
    "orientation_confidence",
    "orientation_applied",
    "perspective_confidence",
    "perspective_corrected",
    "crop_applied",
    "blur_score",
    "brightness",
    "contrast",
    "skew_angle",
    "skew_confidence",
    "deskew_applied",
    "contrast_enhanced",
    "quality_status",
    "quality_flags",
    "sha256",
    "content_ratio",
    "duplicate_group",
    "audit_status",
    "audit_flags",
    "include_in_training",
)
CORE_MANIFEST_FIELDS = MANIFEST_FIELDS[:6]
LOW_CONTENT_THRESHOLD = 0.01


class PreparationError(RuntimeError):
    """Raised when the layout dataset cannot be prepared safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prepare_layout_data",
        description=(
            "Render every document page as normalized PNG and create an "
            "auditable manifest.csv for layout annotation."
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
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Only render/copy pages; skip OpenCV quality normalization",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help=(
            "Recompute audit columns in an existing manifest without rendering "
            "or copying the images again"
        ),
    )
    parser.add_argument(
        "--resume-preprocessing",
        action="store_true",
        help=(
            "Rebuild a missing manifest from already rendered images, then "
            "resume OpenCV preprocessing without rendering PDFs again"
        ),
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help=(
            "With --resume-preprocessing, rerun normalization on every image "
            "instead of skipping pages already present in the manifest"
        ),
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
    filename = f"{output_stem}.png"
    destination = images_dir / filename
    try:
        with Image.open(document) as image:
            # Apply camera/phone EXIF orientation before discarding metadata.
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized.size
            normalized.save(destination, format="PNG")
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


def _empty_quality_fields() -> dict[str, str]:
    return {
        "orientation_angle": "",
        "orientation_confidence": "",
        "orientation_applied": "no",
        "perspective_confidence": "",
        "perspective_corrected": "no",
        "crop_applied": "no",
        "blur_score": "",
        "brightness": "",
        "contrast": "",
        "skew_angle": "",
        "skew_confidence": "",
        "deskew_applied": "no",
        "contrast_enhanced": "no",
        "quality_status": "not_run",
        "quality_flags": "",
    }


def _initialize_quality_fields(rows: list[dict[str, object]]) -> None:
    for row in rows:
        for name, value in _empty_quality_fields().items():
            row.setdefault(name, value)


def _normalize_rows(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    preprocess: bool,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    orientation_classifier=None,
) -> None:
    """Normalize final images and attach quality information to each row."""

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        image_path = _resolve_manifest_image(output_dir, row.get("image"))
        has_current_geometric_audit = bool(row.get("orientation_confidence")) and bool(
            row.get("perspective_confidence")
        )
        if (
            resume
            and has_current_geometric_audit
            and row.get("quality_status") in {"accepted", "corrected", "review"}
        ):
            continue
        if preprocess:
            try:
                quality = normalize_image(
                    image_path,
                    orientation_classifier=orientation_classifier,
                )
            except Exception as exc:
                raise PreparationError(
                    f"Cannot preprocess image {image_path}: {exc}"
                ) from exc
            row.update(quality.to_manifest_fields())
        else:
            row.update(_empty_quality_fields())
        with Image.open(image_path) as image:
            row["width"], row["height"] = image.size
        if preprocess:
            print(
                f"  Normalize [{index}/{total}] {image_path.name}: "
                f"{row['quality_status']}"
            )
        if checkpoint_path is not None and index % 25 == 0:
            _write_manifest(rows, checkpoint_path)

    if checkpoint_path is not None:
        _write_manifest(rows, checkpoint_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_ratio(path: Path) -> float:
    """Return the proportion of pixels darker than near-white."""

    try:
        with Image.open(path) as image:
            if "A" in image.getbands():
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                grayscale = background.convert("L")
            else:
                grayscale = image.convert("L")
            histogram = grayscale.histogram()
            pixel_count = grayscale.width * grayscale.height
    except Exception as exc:
        raise PreparationError(f"Cannot audit image {path}: {exc}") from exc
    return sum(histogram[:245]) / pixel_count


def _resolve_manifest_image(output_dir: Path, relative_image: object) -> Path:
    if not isinstance(relative_image, str) or not relative_image:
        raise PreparationError("Manifest contains an invalid image path.")
    output_root = output_dir.resolve()
    image_path = (output_root / Path(relative_image)).resolve()
    if not image_path.is_relative_to(output_root):
        raise PreparationError(
            f"Manifest image escapes the dataset directory: {relative_image}"
        )
    if not image_path.is_file():
        raise PreparationError(f"Manifest image does not exist: {image_path}")
    return image_path


def _audit_rows(
    rows: list[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    """Enrich manifest rows with non-destructive audit metadata."""

    hashes: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        image_path = _resolve_manifest_image(output_dir, row.get("image"))
        digest = _sha256(image_path)
        ratio = _content_ratio(image_path)
        row["sha256"] = digest
        row["content_ratio"] = f"{ratio:.6f}"
        hashes.setdefault(digest, []).append(index)

    duplicate_groups: dict[int, str] = {}
    duplicate_sets = sorted(
        (indexes for indexes in hashes.values() if len(indexes) > 1),
        key=lambda indexes: indexes[0],
    )
    for group_number, indexes in enumerate(duplicate_sets, start=1):
        group_name = f"dup_{group_number:03d}"
        for index in indexes:
            duplicate_groups[index] = group_name

    for index, row in enumerate(rows):
        flags: list[str] = []
        duplicate_group = duplicate_groups.get(index, "")
        if duplicate_group:
            flags.append("exact_duplicate")
        if float(row["content_ratio"]) < LOW_CONTENT_THRESHOLD:
            flags.append("low_content")
        row["duplicate_group"] = duplicate_group
        quality_requires_review = row.get("quality_status") == "review"
        row["audit_status"] = "review" if flags or quality_requires_review else "keep"
        row["audit_flags"] = ";".join(flags)
        # Valid multipage documents are never excluded automatically, and an
        # existing human decision survives a later manifest refresh.
        existing_decision = str(row.get("include_in_training", "")).lower()
        row["include_in_training"] = (
            existing_decision if existing_decision in {"yes", "no"} else "yes"
        )
    return rows


def _write_manifest(rows: list[dict[str, object]], manifest_path: Path) -> None:
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        with temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_manifest(manifest_path: Path) -> list[dict[str, object]]:
    try:
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            fieldnames = set(reader.fieldnames or ())
            missing = [
                name for name in CORE_MANIFEST_FIELDS if name not in fieldnames
            ]
            if missing:
                raise PreparationError(
                    f"Manifest is missing field(s): {', '.join(missing)}"
                )
            rows = [
                {name: row.get(name, "") for name in MANIFEST_FIELDS}
                for row in reader
            ]
    except FileNotFoundError as exc:
        raise PreparationError(f"Manifest does not exist: {manifest_path}") from exc
    if not rows:
        raise PreparationError(f"Manifest contains no image: {manifest_path}")
    _initialize_quality_fields(rows)
    return rows


def _recover_rows_from_rendered_images(
    input_dir: Path, output_dir: Path
) -> list[dict[str, object]]:
    """Reconstruct manifest rows without rendering source documents again."""

    source_root = input_dir.expanduser().resolve()
    destination_root = output_dir.expanduser().resolve()
    images_dir = destination_root / "images"
    if not images_dir.is_dir():
        raise PreparationError(
            f"Rendered images directory does not exist: {images_dir}"
        )

    documents = _discover_documents(source_root)
    rows: list[dict[str, object]] = []
    expected_paths: set[Path] = set()
    used_stems: set[str] = set()
    for document in documents:
        relative_source = document.relative_to(source_root).as_posix()
        output_stem = _unique_stem(document, source_root, used_stems)
        if document.suffix.lower() == ".pdf":
            try:
                with pymupdf.open(document) as pdf:
                    page_count = pdf.page_count
            except Exception as exc:
                raise PreparationError(
                    f"Cannot inspect PDF {document}: {exc}"
                ) from exc
            filenames = [
                f"{output_stem}__p{page_number:03d}.png"
                for page_number in range(1, page_count + 1)
            ]
            source_type = "pdf"
        else:
            filenames = [f"{output_stem}.png"]
            source_type = "image"

        for page_number, filename in enumerate(filenames, start=1):
            image_path = (images_dir / filename).resolve()
            if not image_path.is_file():
                raise PreparationError(
                    f"Cannot resume; expected rendered image is missing: {image_path}"
                )
            expected_paths.add(image_path)
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
            except Exception as exc:
                raise PreparationError(
                    f"Cannot inspect rendered image {image_path}: {exc}"
                ) from exc
            row: dict[str, object] = {
                "image": f"images/{filename}",
                "source_document": relative_source,
                "page": page_number,
                "source_type": source_type,
                "width": width,
                "height": height,
                "include_in_training": "yes",
            }
            row.update(_empty_quality_fields())
            rows.append(row)

    discovered_paths = {
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if discovered_paths != expected_paths:
        unexpected = sorted(path.name for path in discovered_paths - expected_paths)
        details = (
            f"; unexpected: {', '.join(unexpected[:10])}" if unexpected else ""
        )
        raise PreparationError(
            "Cannot resume because rendered images do not match source documents"
            + details
        )
    return rows


def resume_preprocessing(
    input_dir: Path,
    output_dir: Path,
    *,
    orientation_classifier=None,
    force_reprocess: bool = False,
) -> list[dict[str, object]]:
    """Resume normalization from a checkpoint or reconstruct its manifest."""

    destination_root = output_dir.expanduser().resolve()
    manifest_path = destination_root / "manifest.csv"
    if manifest_path.is_file():
        rows = _read_manifest(manifest_path)
    else:
        print("Manifest missing; rebuilding it from source documents and images...")
        rows = _recover_rows_from_rendered_images(input_dir, destination_root)
        _write_manifest(rows, manifest_path)

    _normalize_rows(
        rows,
        destination_root,
        preprocess=True,
        checkpoint_path=manifest_path,
        resume=not force_reprocess,
        orientation_classifier=orientation_classifier,
    )
    _audit_rows(rows, destination_root)
    _write_manifest(rows, manifest_path)
    return rows


def refresh_manifest(output_dir: Path) -> list[dict[str, object]]:
    """Audit existing images and update their manifest without reconversion."""

    destination_root = output_dir.expanduser().resolve()
    manifest_path = destination_root / "manifest.csv"
    rows = _read_manifest(manifest_path)
    _audit_rows(rows, destination_root)
    _write_manifest(rows, manifest_path)
    return rows


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    dpi: int = 200,
    *,
    preprocess: bool = True,
    orientation_classifier=None,
) -> list[dict[str, object]]:
    """Prepare page images without modifying any source document."""

    if dpi <= 0:
        raise PreparationError("DPI must be a positive integer.")

    source_root = input_dir.expanduser().resolve()
    destination_root = output_dir.expanduser().resolve()
    documents = _discover_documents(source_root)
    images_dir, manifest_path = _ensure_empty_destination(destination_root)

    rows: list[dict[str, object]] = []
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
            print("  Image converted")

    _initialize_quality_fields(rows)
    # Write a recoverable checkpoint before the potentially long OpenCV pass.
    _write_manifest(rows, manifest_path)
    _normalize_rows(
        rows,
        destination_root,
        preprocess=preprocess,
        checkpoint_path=manifest_path,
        orientation_classifier=orientation_classifier,
    )
    _audit_rows(rows, destination_root)
    _write_manifest(rows, manifest_path)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.force_reprocess and not args.resume_preprocessing:
            raise PreparationError(
                "--force-reprocess must be used with --resume-preprocessing."
            )
        if args.resume_preprocessing:
            if args.refresh_manifest or args.no_preprocess:
                raise PreparationError(
                    "--resume-preprocessing cannot be combined with "
                    "--refresh-manifest or --no-preprocess."
                )
            orientation_service = DocumentOrientationService()
            rows = resume_preprocessing(
                args.input,
                args.output,
                orientation_classifier=orientation_service.predict,
                force_reprocess=args.force_reprocess,
            )
            review_count = sum(row["audit_status"] == "review" for row in rows)
            print(
                f"Preprocessing complete: {len(rows)} image(s), "
                f"{review_count} page(s) to review -> "
                f"{(args.output / 'manifest.csv').resolve()}"
            )
            return 0
        if args.refresh_manifest:
            rows = refresh_manifest(args.output)
            review_count = sum(row["audit_status"] == "review" for row in rows)
            print(
                f"Manifest refreshed: {len(rows)} image(s), "
                f"{review_count} page(s) to review -> "
                f"{(args.output / 'manifest.csv').resolve()}"
            )
            return 0

        orientation_classifier = None
        if not args.no_preprocess:
            orientation_classifier = DocumentOrientationService().predict
        rows = prepare_dataset(
            args.input,
            args.output,
            args.dpi,
            preprocess=not args.no_preprocess,
            orientation_classifier=orientation_classifier,
        )
        pdf_pages = sum(row["source_type"] == "pdf" for row in rows)
        copied_images = len(rows) - pdf_pages
        review_count = sum(row["audit_status"] == "review" for row in rows)
        corrected_count = sum(row["quality_status"] == "corrected" for row in rows)
        print(
            f"Dataset ready: {len(rows)} image(s) "
            f"({pdf_pages} PDF page(s), {copied_images} copied image(s)), "
            f"{corrected_count} corrected, {review_count} page(s) to review "
            f"-> {args.output.resolve()}"
        )
        return 0
    except PreparationError as exc:
        print(f"Preparation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
