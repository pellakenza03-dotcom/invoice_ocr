"""OCR integration layer."""

from .pipeline import OCRConfig, PaddleOCRService, get_ocr_service

__all__ = ["OCRConfig", "PaddleOCRService", "get_ocr_service"]
