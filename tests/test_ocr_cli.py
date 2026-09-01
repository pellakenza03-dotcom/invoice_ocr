from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ocr.cli import discover_documents, main


class FakeLayoutResult:
    def __init__(self):
        self.image_saved_to: str | None = None

    def save_to_img(self, output_path: str):
        self.image_saved_to = output_path


class FakeResult(dict):
    def __init__(self):
        self.json_saved_to: str | None = None
        self.layout_result = FakeLayoutResult()
        super().__init__(layout_det_res=self.layout_result)

    def save_to_json(self, output_dir: str):
        self.json_saved_to = output_dir


class FakeService:
    def __init__(self, result: FakeResult):
        self.result = result
        self.input_path: Path | None = None
        self.input_paths: list[Path] = []

    def predict(self, input_path: Path):
        self.input_path = input_path
        self.input_paths.append(input_path)
        return [self.result]


class OCRCLITests(unittest.TestCase):
    def test_runs_ocr_and_saves_native_json(self):
        result = FakeResult()
        service = FakeService(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "invoice.pdf"
            output_dir = root / "output"
            document.touch()
            exit_code = main(
                [str(document)],
                service_factory=lambda _config: service,
                output_dir_override=output_dir,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.input_path, document.resolve())
        self.assertEqual(result.json_saved_to, str(output_dir))
        self.assertEqual(
            result.layout_result.image_saved_to,
            str(output_dir / "invoice_0_layout_det_res.png"),
        )

    def test_discovers_supported_documents_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            (input_dir / "z.png").touch()
            (input_dir / "A.pdf").touch()
            (input_dir / "ignored.txt").touch()

            documents = discover_documents(input_dir)

        self.assertEqual([path.name for path in documents], ["A.pdf", "z.png"])

    def test_batch_reuses_service_and_applies_low_threshold_profile(self):
        result = FakeResult()
        service = FakeService(result)
        received_configs = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "b.png").touch()
            (input_dir / "a.pdf").touch()

            exit_code = main(
                [str(input_dir), "--audit-profile", "low_threshold"],
                service_factory=lambda config: (
                    received_configs.append(config) or service
                ),
                output_dir_override=output_dir,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(received_configs), 1)
        self.assertEqual(received_configs[0].layout_threshold, 0.3)
        self.assertEqual(received_configs[0].layout_merge_bboxes_mode, "union")
        self.assertEqual(
            [path.name for path in service.input_paths], ["a.pdf", "b.png"]
        )


if __name__ == "__main__":
    unittest.main()
