from __future__ import annotations

import unittest
from pathlib import Path

from src.ocr.cli import main


class FakeResult:
    def __init__(self):
        self.saved_to: str | None = None

    def save_to_json(self, output_dir: str):
        self.saved_to = output_dir


class FakeService:
    def __init__(self, result: FakeResult):
        self.result = result
        self.input_path: Path | None = None

    def predict(self, input_path: Path):
        self.input_path = input_path
        return [self.result]


class OCRCLITests(unittest.TestCase):
    def test_runs_ocr_and_saves_native_json(self):
        result = FakeResult()
        service = FakeService(result)

        exit_code = main(
            ["invoice.pdf"],
            service_factory=lambda _config: service,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.input_path, Path("invoice.pdf"))
        self.assertEqual(result.saved_to, str(Path("data/ocr")))


if __name__ == "__main__":
    unittest.main()
