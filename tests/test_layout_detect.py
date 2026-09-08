from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from layout_detect.cli import discover_images, load_profile, main
from ocr.pipeline import DEFAULT_CONFIG_PATH, OCRConfig, OCRConfigurationError


class FakeLayoutResult(dict):
    def __init__(self):
        super().__init__(boxes=[{"label": "table"}, {"label": "text"}])
        self.json_path = None
        self.image_path = None

    def save_to_json(self, path):
        self.json_path = path

    def save_to_img(self, path):
        self.image_path = path


class FakeService:
    def __init__(self):
        self.paths = []
        self.results = []

    def predict(self, path):
        self.paths.append(path)
        layout_result = FakeLayoutResult()
        self.results.append(layout_result)
        return [{"layout_det_res": layout_result}]


def write_config(path: Path) -> None:
    path.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")


class LayoutDetectTests(unittest.TestCase):
    def test_discovers_images_only_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.JPG").touch()
            (root / "A.png").touch()
            (root / "ignored.pdf").touch()
            self.assertEqual(
                [path.name for path in discover_images(root)], ["A.png", "z.JPG"]
            )

    def test_batch_loads_one_service_and_saves_each_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            output = root / "output"
            config_path = root / "ocr.json"
            inputs.mkdir()
            (inputs / "b.jpg").touch()
            (inputs / "a.png").touch()
            write_config(config_path)
            service = FakeService()
            received_configs = []

            exit_code = main(
                [str(inputs), "--profile", "low_threshold", "--config", str(config_path)],
                service_factory=lambda config: (
                    received_configs.append(config) or service
                ),
                output_dir_override=output,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(received_configs), 1)
            self.assertEqual(
                received_configs[0].layout_threshold,
                load_profile("low_threshold")["layout_threshold"],
            )
            self.assertEqual(received_configs[0].layout_merge_bboxes_mode, "union")
            self.assertEqual([path.name for path in service.paths], ["a.png", "b.jpg"])
            self.assertEqual(
                service.results[0].json_path, str(output / "a.layout.json")
            )
            self.assertEqual(
                service.results[0].image_path, str(output / "a.layout.png")
            )

    def test_shared_config_requires_layout_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(OCRConfigurationError, "missing"):
                OCRConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
