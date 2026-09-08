from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ocr.cli import (
    _page_outputs_are_complete,
    discover_documents,
    load_audit_overrides,
    main,
)


class FakeLayoutResult:
    def __init__(self):
        self.image_saved_to: str | None = None
        self.json_saved_to: str | None = None

    def save_to_json(self, output_path: str):
        self.json_saved_to = output_path

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
    def test_resume_requires_all_valid_page_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "invoice.png"
            image.touch()
            (root / "invoice.structure.json").write_text(
                '{"layout_det_res": {}}', encoding="utf-8"
            )
            (root / "invoice.layout.json").write_text(
                '{"boxes": []}', encoding="utf-8"
            )
            (root / "invoice.layout.png").write_bytes(b"png")

            self.assertTrue(_page_outputs_are_complete(image, root))
            (root / "invoice.layout.json").write_text("invalid", encoding="utf-8")
            self.assertFalse(_page_outputs_are_complete(image, root))

    def test_resume_skips_completed_page_without_loading_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "images"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            image = input_dir / "invoice.png"
            image.touch()
            (output_dir / "invoice.structure.json").write_text(
                '{"layout_det_res": {}}', encoding="utf-8"
            )
            (output_dir / "invoice.layout.json").write_text(
                '{"boxes": []}', encoding="utf-8"
            )
            (output_dir / "invoice.layout.png").write_bytes(b"png")

            exit_code = main(
                [str(input_dir), "--resume"],
                service_factory=lambda _config: self.fail("pipeline was loaded"),
                output_dir_override=output_dir,
            )

            self.assertEqual(exit_code, 0)

    def test_runs_ocr_and_saves_native_json(self):
        result = FakeResult()
        service = FakeService(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "invoice.pdf"
            output_dir = root / "output"
            document.touch()
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [str(document)],
                    service_factory=lambda _config: service,
                    output_dir_override=output_dir,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(service.input_path, document.resolve())
        self.assertEqual(
            result.json_saved_to,
            str(output_dir / "invoice__p001.structure.json"),
        )
        self.assertEqual(
            result.layout_result.json_saved_to,
            str(output_dir / "invoice__p001.layout.json"),
        )
        self.assertEqual(
            result.layout_result.image_saved_to,
            str(output_dir / "invoice__p001.layout.png"),
        )
        self.assertIn("[1/1] Starting: invoice.pdf", stdout.getvalue())
        self.assertIn("[1/1] Finished: invoice.pdf", stdout.getvalue())

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
        self.assertEqual(
            received_configs[0].layout_threshold,
            load_audit_overrides("low_threshold")["layout_threshold"],
        )
        self.assertEqual(received_configs[0].layout_merge_bboxes_mode, "union")
        self.assertEqual(
            [path.name for path in service.input_paths], ["a.pdf", "b.png"]
        )


if __name__ == "__main__":
    unittest.main()
