from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ocr.pipeline import (
    OCRConfig,
    OCRConfigurationError,
    OCRInputError,
    PaddleOCRService,
)


class FakePipeline:
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        self.kwargs = kwargs
        self.inputs: list[str] = []

    def predict(self, document_path: str):
        self.inputs.append(document_path)
        return [{"page_index": 0}]


class OCRPipelineTests(unittest.TestCase):
    def setUp(self):
        FakePipeline.instances = 0

    def test_configuration_is_forwarded_and_pipeline_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "invoice.pdf"
            document.touch()
            service = PaddleOCRService(
                OCRConfig.from_json(), pipeline_factory=FakePipeline
            )

            first = service.predict(document)
            second = service.predict(document)

        self.assertEqual(FakePipeline.instances, 1)
        self.assertEqual(first, [{"page_index": 0}])
        self.assertEqual(second, first)
        self.assertEqual(service._pipeline.kwargs["device"], "cpu")
        self.assertFalse(service._pipeline.kwargs["enable_mkldnn"])
        self.assertEqual(service._pipeline.kwargs["layout_threshold"], 0.5)
        self.assertTrue(service._pipeline.kwargs["layout_nms"])
        self.assertEqual(
            service._pipeline.kwargs["layout_merge_bboxes_mode"], "large"
        )

    def test_rejects_missing_and_unsupported_inputs(self):
        service = PaddleOCRService(
            OCRConfig.from_json(), pipeline_factory=FakePipeline
        )
        with self.assertRaises(OCRInputError):
            service.predict("missing.pdf")

        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "invoice.txt"
            document.touch()
            with self.assertRaises(OCRInputError):
                service.predict(document)

    def test_rejects_unknown_configuration_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "ocr.json"
            config_path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaises(OCRConfigurationError):
                OCRConfig.from_json(config_path)

    def test_rejects_missing_configuration_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "ocr.json"
            config_path.write_text(json.dumps({"device": "cpu"}), encoding="utf-8")
            with self.assertRaises(OCRConfigurationError):
                OCRConfig.from_json(config_path)

    def test_rejects_invalid_layout_overrides(self):
        config = OCRConfig.from_json()
        with self.assertRaises(OCRConfigurationError):
            config.with_overrides({"layout_threshold": 1.5})
        with self.assertRaises(OCRConfigurationError):
            config.with_overrides({"layout_merge_bboxes_mode": "invalid"})


if __name__ == "__main__":
    unittest.main()
