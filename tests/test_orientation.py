import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing.orientation import DocumentOrientationService


class _FakeOrientationModel:
    def predict(self, candidates, batch_size):
        assert len(candidates) == 4
        assert batch_size == 4
        upright_scores = (0.05, 0.96, 0.03, 0.04)
        for upright_score in upright_scores:
            yield {
                "label_names": ["0", "90", "180", "270"],
                "scores": [upright_score, 0.01, 0.01, 0.01],
            }


class OrientationTests(unittest.TestCase):
    def test_selects_variant_with_highest_upright_probability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sideways.png"
            cv2.imwrite(
                str(path), np.full((200, 400, 3), 255, dtype=np.uint8)
            )
            service = DocumentOrientationService(
                model_factory=lambda **kwargs: _FakeOrientationModel()
            )

            angle, confidence = service.predict(path)

            self.assertEqual(angle, 90)
            self.assertAlmostEqual(confidence, 0.96)


if __name__ == "__main__":
    unittest.main()
