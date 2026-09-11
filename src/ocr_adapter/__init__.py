"""Convert native PP-StructureV3 exports to the canonical OCR schema."""

from .adapter import (
    AdapterError,
    adapt_document,
    adapt_page,
    discover_document_groups,
    validate_canonical_document,
)

__all__ = [
    "AdapterError",
    "adapt_document",
    "adapt_page",
    "discover_document_groups",
    "validate_canonical_document",
]
