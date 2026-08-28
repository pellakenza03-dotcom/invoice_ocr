"""Allow ``python -m src.ocr`` to run the OCR CLI."""

from .cli import main


raise SystemExit(main())
