import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preprocessing import normalize_image


class PreprocessingTests(unittest.TestCase):
    def test_audits_clean_page_without_changing_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "page.png"
            image = np.full((600, 400, 3), 255, dtype=np.uint8)
            for y in range(100, 500, 35):
                cv2.line(image, (60, y), (340, y), (0, 0, 0), 2)
            cv2.imwrite(str(path), image)

            result = normalize_image(path)

            self.assertFalse(result.deskew_applied)
            self.assertGreater(result.blur_score, 0)
            normalized = cv2.imread(str(path))
            self.assertEqual(normalized.shape[:2], (600, 400))

    def test_enhances_only_a_severely_low_contrast_page(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "page.png"
            image = np.full((300, 200, 3), 130, dtype=np.uint8)
            cv2.line(image, (20, 100), (180, 100), (120, 120, 120), 2)
            cv2.imwrite(str(path), image)

            result = normalize_image(path)

            self.assertTrue(result.contrast_enhanced)
            self.assertIn("contrast_enhanced", result.quality_flags)

    def test_applies_high_confidence_right_angle_orientation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sideways.png"
            image = np.full((200, 400, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                "INVOICE",
                (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )
            cv2.imwrite(str(path), image)

            result = normalize_image(
                path, orientation_classifier=lambda _: (90, 0.98)
            )

            self.assertTrue(result.orientation_applied)
            self.assertEqual(result.orientation_angle, 90)
            rotated = cv2.imread(str(path))
            self.assertEqual(rotated.shape[:2], (400, 200))

    def test_crops_and_rectifies_a_dominant_document_quad(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "photo.png"
            image = np.full((900, 700, 3), 35, dtype=np.uint8)
            document = np.array(
                [[90, 80], [630, 120], [600, 830], [60, 790]], dtype=np.int32
            )
            cv2.fillConvexPoly(image, document, (245, 245, 245))
            cv2.polylines(image, [document], True, (255, 255, 255), 5)
            cv2.imwrite(str(path), image)

            result = normalize_image(path)

            self.assertTrue(result.perspective_corrected)
            self.assertTrue(result.crop_applied)
            cropped = cv2.imread(str(path))
            self.assertLess(cropped.shape[0], 900)
            self.assertLess(cropped.shape[1], 700)

    def test_does_not_mistake_a_large_table_for_the_document_boundary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invoice.png"
            image = np.full((900, 700, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (60, 170), (640, 800), (0, 0, 0), 4)
            for y in range(250, 800, 70):
                cv2.line(image, (60, y), (640, y), (0, 0, 0), 2)
            cv2.imwrite(str(path), image)

            result = normalize_image(path)

            self.assertFalse(result.crop_applied)
            unchanged = cv2.imread(str(path))
            self.assertEqual(unchanged.shape[:2], (900, 700))

    def test_crops_paper_inside_a_letterboxed_phone_photo(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "letterboxed.png"
            image = np.full((1200, 800, 3), 255, dtype=np.uint8)
            image[120:1080, 80:720] = (10, 10, 10)
            image[250:970, 80:720] = (65, 65, 65)
            document = np.array(
                [[155, 330], [650, 350], [665, 900], [125, 885]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(image, document, (235, 235, 235))
            for y in range(420, 830, 45):
                cv2.line(image, (175, y), (625, y + 5), (80, 80, 80), 2)
            cv2.imwrite(str(path), image)

            result = normalize_image(path)

            self.assertTrue(result.crop_applied)
            cropped = cv2.imread(str(path))
            self.assertLess(cropped.shape[0], 700)
            self.assertLess(cropped.shape[1], 600)

    def test_corrects_a_high_confidence_small_skew_without_cropping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "skewed.png"
            image = np.full((800, 600, 3), 255, dtype=np.uint8)
            for y in range(100, 700, 40):
                cv2.line(image, (80, y), (520, y + 16), (0, 0, 0), 2)
            cv2.imwrite(str(path), image)

            result = normalize_image(path)

            self.assertTrue(result.deskew_applied)
            self.assertGreater(result.skew_confidence, 0.65)
            normalized = cv2.imread(str(path))
            self.assertGreaterEqual(normalized.shape[0], 800)
            self.assertGreaterEqual(normalized.shape[1], 600)


if __name__ == "__main__":
    unittest.main()
