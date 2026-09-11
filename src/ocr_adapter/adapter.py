"""Adapter from PaddleOCR's native PP-StructureV3 JSON to canonical OCR JSON."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "ocr_document.schema.json"
PAGE_EXPORT_PATTERN = re.compile(
    r"^(?P<document>.+)__p(?P<page>0*[1-9][0-9]*)\.structure\.json$",
    re.IGNORECASE,
)


class AdapterError(ValueError):
    """Raised when a native result cannot be converted safely."""


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _confidence(value: object, default: float = 0.0) -> float:
    number = _finite_number(value)
    if number is None:
        return default
    return min(1.0, max(0.0, number))


def _bbox(value: object, width: int, height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_finite_number(item) for item in value]
    if any(item is None for item in numbers):
        return None
    x1, y1, x2, y2 = (float(item) for item in numbers)
    x1, x2 = sorted((min(width, max(0.0, x1)), min(width, max(0.0, x2))))
    y1, y2 = sorted((min(height, max(0.0, y1)), min(height, max(0.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _polygon(value: object, width: int, height: int) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x = _finite_number(point[0])
        y = _finite_number(point[1])
        if x is None or y is None:
            return None
        points.append([min(width, max(0.0, x)), min(height, max(0.0, y))])
    return points


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _source_type(label: object) -> str:
    normalized = str(label or "").strip().lower()
    mapping = {
        "doc_title": "title",
        "document_title": "title",
        "title": "title",
        "header": "header",
        "footer": "footer",
        "table": "table",
        "image": "image",
        "header_image": "image",
        "footer_image": "image",
        "seal": "seal",
        "formula": "formula",
        "chart": "chart",
    }
    return mapping.get(normalized, "text")


def _layout_regions(
    native: dict[str, Any], width: int, height: int, max_table_page_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    raw_boxes = native.get("layout_det_res", {}).get("boxes", [])
    if not isinstance(raw_boxes, list):
        raw_boxes = []
    regions: list[dict[str, Any]] = []
    valid_tables: list[dict[str, Any]] = []
    warnings: list[str] = []
    page_area = width * height

    for index, raw in enumerate(raw_boxes):
        if not isinstance(raw, dict):
            warnings.append(f"Ignored invalid layout region at index {index}.")
            continue
        box = _bbox(raw.get("coordinate"), width, height)
        if box is None:
            warnings.append(f"Ignored invalid layout bounding box at index {index}.")
            continue
        label = str(raw.get("label") or "unknown").strip().lower()
        region = {
            "source_index": index,
            "label": label,
            "type": _source_type(label),
            "bbox": box,
            "score": _confidence(raw.get("score")),
        }
        ratio = _area(box) / page_area
        if region["type"] == "table" and ratio > max_table_page_ratio:
            region["suspicious"] = True
            warnings.append(
                "Ignored suspicious table structure covering "
                f"{ratio:.1%} of the page; all OCR text was preserved."
            )
        else:
            region["suspicious"] = False
            if region["type"] == "table":
                valid_tables.append(region)
        regions.append(region)
    return regions, valid_tables, warnings


def _best_region(
    box: Sequence[float], regions: Iterable[dict[str, Any]], minimum_coverage: float
) -> dict[str, Any] | None:
    candidates: list[tuple[float, float, float, dict[str, Any]]] = []
    box_area = _area(box)
    if box_area <= 0:
        return None
    for region in regions:
        if region["suspicious"]:
            continue
        coverage = _intersection_area(box, region["bbox"]) / box_area
        if coverage >= minimum_coverage:
            candidates.append(
                (coverage, -_area(region["bbox"]), region["score"], region)
            )
    return max(candidates, default=(0.0, 0.0, 0.0, None))[3]


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[dict[str, Any]] = []
        self.row = -1
        self.column = 0
        self._cell: dict[str, Any] | None = None
        self._text: list[str] = []
        self._occupied: set[tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row += 1
            self.column = 0
            return
        if tag not in {"td", "th"}:
            return
        if self.row < 0:
            self.row = 0
        while (self.row, self.column) in self._occupied:
            self.column += 1
        attributes = dict(attrs)
        try:
            row_span = max(1, int(attributes.get("rowspan") or 1))
            column_span = max(1, int(attributes.get("colspan") or 1))
        except ValueError:
            row_span = column_span = 1
        self._cell = {
            "row": self.row,
            "column": self.column,
            "row_span": row_span,
            "column_span": column_span,
        }
        self._text = []
        for row in range(self.row, self.row + row_span):
            for column in range(self.column, self.column + column_span):
                if row != self.row or column != self.column:
                    self._occupied.add((row, column))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() not in {"td", "th"} or self._cell is None:
            return
        text = " ".join("".join(self._text).split())
        self._cell["text"] = text
        self.cells.append(self._cell)
        self.column += self._cell["column_span"]
        self._cell = None
        self._text = []


def _parse_html_cells(html: str) -> list[dict[str, Any]]:
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return []
    return parser.cells


def _cell_evidence(
    cell_box: Sequence[float], blocks: Sequence[dict[str, Any]]
) -> tuple[list[str], float]:
    matches: list[tuple[dict[str, Any], float]] = []
    for block in blocks:
        confidence = _finite_number(block.get("ocr_confidence"))
        if confidence is None:
            # Visual layout-only blocks have geometry but are not OCR evidence.
            continue
        block_box = block["bbox"]
        block_area = _area(block_box)
        if block_area and _intersection_area(cell_box, block_box) / block_area >= 0.5:
            matches.append((block, _confidence(confidence)))
    ids = [block["block_id"] for block, _ in matches]
    mean_confidence = fmean(confidence for _, confidence in matches) if matches else 0.0
    return ids, mean_confidence


def _canonical_tables(
    native: dict[str, Any],
    valid_regions: Sequence[dict[str, Any]],
    blocks: Sequence[dict[str, Any]],
    width: int,
    height: int,
    page_number: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    raw_tables = native.get("table_res_list") or []
    if not isinstance(raw_tables, list):
        raw_tables = []
    if len(raw_tables) != len(valid_regions):
        if raw_tables or valid_regions:
            warnings.append(
                "Table result/layout count mismatch: "
                f"{len(raw_tables)} result(s), {len(valid_regions)} trusted region(s)."
            )

    tables: list[dict[str, Any]] = []
    for table_index, (raw, region) in enumerate(zip(raw_tables, valid_regions)):
        if not isinstance(raw, dict):
            continue
        html = raw.get("pred_html")
        html = html if isinstance(html, str) else ""
        parsed_cells = _parse_html_cells(html)
        raw_cell_boxes = raw.get("cell_box_list") or []
        cell_boxes = [
            box
            for value in raw_cell_boxes
            if (box := _bbox(value, width, height)) is not None
        ]
        if len(parsed_cells) != len(cell_boxes):
            warnings.append(
                f"Table {table_index + 1} has {len(parsed_cells)} HTML cell(s) "
                f"but {len(cell_boxes)} valid cell box(es); only matched cells were kept."
            )
        cells: list[dict[str, Any]] = []
        for parsed, box in zip(parsed_cells, cell_boxes):
            source_ids, confidence = _cell_evidence(box, blocks)
            cell = {
                **parsed,
                "bbox": box,
                "ocr_confidence": confidence,
            }
            if source_ids:
                cell["source_block_ids"] = source_ids
            cells.append(cell)
        table = {
            "table_id": f"p{page_number}_t{table_index}",
            "bbox": region["bbox"],
            "structure_confidence": region["score"],
            "cells": cells,
        }
        if html:
            table["html"] = html
        tables.append(table)
    return tables


def adapt_page(
    native: dict[str, Any],
    page_number: int,
    *,
    minimum_layout_coverage: float = 0.5,
    max_table_page_ratio: float = 0.75,
) -> dict[str, Any]:
    """Convert one native PP-StructureV3 page export."""

    width = native.get("width")
    height = native.get("height")
    if not isinstance(width, int) or width < 1 or not isinstance(height, int) or height < 1:
        raise AdapterError("Native page width and height must be positive integers.")
    if page_number < 1:
        raise AdapterError("Page number must be positive.")

    regions, valid_tables, warnings = _layout_regions(
        native, width, height, max_table_page_ratio
    )
    ocr = native.get("overall_ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    scores = ocr.get("rec_scores") or []
    boxes = ocr.get("rec_boxes") or []
    polygons = ocr.get("rec_polys") or []
    if not all(isinstance(value, list) for value in (texts, scores, boxes, polygons)):
        raise AdapterError("Native overall_ocr_res contains invalid OCR arrays.")
    lengths = {len(texts), len(scores), len(boxes)}
    if len(lengths) != 1:
        warnings.append(
            "OCR array lengths differ; entries without text, confidence or boxes were ignored."
        )

    blocks: list[dict[str, Any]] = []
    replacement_count = 0
    for index, (raw_text, raw_score, raw_box) in enumerate(zip(texts, scores, boxes)):
        text = str(raw_text).strip()
        box = _bbox(raw_box, width, height)
        if not text or box is None:
            warnings.append(f"Ignored invalid OCR entry at index {index}.")
            continue
        if "\ufffd" in text:
            replacement_count += 1
        region = _best_region(box, regions, minimum_layout_coverage)
        block = {
            "block_id": f"p{page_number}_b{len(blocks)}",
            "type": region["type"] if region is not None else "text",
            "text": text,
            "bbox": box,
            "ocr_confidence": _confidence(raw_score),
            "reading_order": len(blocks),
        }
        if index < len(polygons):
            polygon = _polygon(polygons[index], width, height)
            if polygon is not None:
                block["polygon"] = polygon
        blocks.append(block)

    if not blocks:
        warnings.append("No usable OCR text was found on this page.")
    if replacement_count:
        warnings.append(
            f"OCR text contains the Unicode replacement character in "
            f"{replacement_count} block(s)."
        )

    # Retain non-text visual regions even when OCR naturally produced no text for them.
    for region in regions:
        if region["suspicious"] or region["type"] not in {
            "image",
            "seal",
            "formula",
            "chart",
        }:
            continue
        if any(_intersection_area(region["bbox"], block["bbox"]) > 0 for block in blocks):
            continue
        blocks.append(
            {
                "block_id": f"p{page_number}_b{len(blocks)}",
                "type": region["type"],
                "bbox": region["bbox"],
                "reading_order": len(blocks),
            }
        )

    angle = (native.get("doc_preprocessor_res") or {}).get("angle", 0)
    orientation = int(angle) if angle in {0, 90, 180, 270} else 0
    tables = _canonical_tables(
        native,
        valid_tables,
        blocks,
        width,
        height,
        page_number,
        warnings,
    )
    page = {
        "page_number": page_number,
        "width": width,
        "height": height,
        "orientation": orientation,
        "blocks": blocks,
        "tables": tables,
    }
    if warnings:
        page["warnings"] = warnings
    return page


def _load_native(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdapterError(f"Native structure JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"Native structure JSON must be an object: {path}")
    return value


def _page_identity(path: Path) -> tuple[str, int]:
    match = PAGE_EXPORT_PATTERN.match(path.name)
    if match:
        return match.group("document"), int(match.group("page"))
    suffix = ".structure.json"
    if not path.name.lower().endswith(suffix):
        raise AdapterError(f"Expected a *.structure.json file: {path}")
    return path.name[: -len(suffix)], 1


def discover_document_groups(input_path: str | Path) -> dict[str, list[tuple[int, Path]]]:
    """Discover native exports and group ``document__pNNN`` pages."""

    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.glob("*.structure.json"), key=lambda item: item.name.casefold())
    else:
        raise AdapterError(f"Adapter input does not exist: {path}")
    if not paths:
        raise AdapterError(f"No *.structure.json files found in: {path}")

    groups: dict[str, list[tuple[int, Path]]] = {}
    seen: set[tuple[str, int]] = set()
    for item in paths:
        document_id, page_number = _page_identity(item)
        identity = (document_id.casefold(), page_number)
        if identity in seen:
            raise AdapterError(
                f"Duplicate page {page_number} for document {document_id!r}."
            )
        seen.add(identity)
        groups.setdefault(document_id, []).append((page_number, item))
    for pages in groups.values():
        pages.sort(key=lambda value: value[0])
    return groups


def _source_details(natives: Sequence[dict[str, Any]], document_id: str) -> tuple[str, str | None, Path | None]:
    input_paths = [
        Path(value["input_path"])
        for value in natives
        if isinstance(value.get("input_path"), str) and value["input_path"]
    ]
    if input_paths and len({path.name for path in input_paths}) == 1:
        source = input_paths[0]
        return source.name, mimetypes.guess_type(source.name)[0], source if source.is_file() else None
    source_name = document_id
    mime = mimetypes.guess_type(input_paths[0].name)[0] if input_paths else None
    return source_name, mime, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paddleocr_version() -> str | None:
    try:
        return version("paddleocr")
    except PackageNotFoundError:
        return None


def adapt_document(
    pages: Sequence[tuple[int, str | Path]],
    *,
    document_id: str | None = None,
    generated_at: str | None = None,
    minimum_layout_coverage: float = 0.5,
    max_table_page_ratio: float = 0.75,
) -> dict[str, Any]:
    """Convert ordered native page exports into one canonical OCR document."""

    if not pages:
        raise AdapterError("A document must contain at least one page export.")
    normalized = sorted((number, Path(path)) for number, path in pages)
    page_numbers = [number for number, _ in normalized]
    if page_numbers != list(range(1, len(normalized) + 1)):
        raise AdapterError(
            "Document pages must be contiguous and numbered from 1; received "
            + ", ".join(map(str, page_numbers))
            + "."
        )
    natives = [_load_native(path) for _, path in normalized]
    inferred_id = _page_identity(normalized[0][1])[0]
    identity = document_id or inferred_id
    source_file, mime, source_path = _source_details(natives, identity)
    engine: dict[str, Any] = {"name": "PP-StructureV3"}
    installed_version = _paddleocr_version()
    if installed_version:
        engine["version"] = installed_version

    document: dict[str, Any] = {
        "schema_version": "1.0.0",
        "document_id": identity,
        "source_file": source_file,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ocr_engine": engine,
        "page_count": len(normalized),
        "pages": [
            adapt_page(
                native,
                number,
                minimum_layout_coverage=minimum_layout_coverage,
                max_table_page_ratio=max_table_page_ratio,
            )
            for (number, _), native in zip(normalized, natives)
        ],
    }
    if mime:
        document["source_mime_type"] = mime
    if source_path is not None:
        document["source_sha256"] = _sha256(source_path)
    warnings = [
        f"Page {page['page_number']}: {warning}"
        for page in document["pages"]
        for warning in page.get("warnings", [])
    ]
    if warnings:
        document["warnings"] = warnings
    return document


def validate_canonical_document(
    document: dict[str, Any], schema_path: str | Path = DEFAULT_SCHEMA_PATH
) -> None:
    """Validate a converted document against the canonical JSON Schema."""

    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise AdapterError("jsonschema is required to validate canonical OCR output.") from exc
    schema_file = Path(schema_path)
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise AdapterError(f"Cannot load canonical OCR schema {schema_file}: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"/{'/'.join(map(str, error.absolute_path))}: {error.message}"
            for error in errors[:10]
        )
        raise AdapterError(f"Canonical OCR validation failed: {details}")
