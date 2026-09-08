"""Measure page quality and apply only high-confidence corrections."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

MAX_ANALYSIS_DIMENSION = 1000
MIN_DESKEW_ANGLE = 0.5
MAX_DESKEW_ANGLE = 5.0
MIN_DESKEW_CONFIDENCE = 0.65
BLUR_REVIEW_THRESHOLD = 45.0
DARK_REVIEW_THRESHOLD = 65.0
BRIGHT_REVIEW_THRESHOLD = 250.0
LOW_CONTRAST_THRESHOLD = 22.0
MIN_ORIENTATION_CONFIDENCE = 0.85
MIN_DOCUMENT_AREA_RATIO = 0.15
MAX_DOCUMENT_AREA_RATIO = 0.94
MIN_PERSPECTIVE_CONFIDENCE = 0.85


@dataclass(frozen=True, slots=True)
class ImageQualityResult:
    """Quality measurements and transformations applied to one page."""

    orientation_angle: int
    orientation_confidence: float
    orientation_applied: bool
    perspective_confidence: float
    perspective_corrected: bool
    crop_applied: bool
    blur_score: float
    brightness: float
    contrast: float
    skew_angle: float
    skew_confidence: float
    deskew_applied: bool
    contrast_enhanced: bool
    quality_status: str
    quality_flags: str

    def to_manifest_fields(self) -> dict[str, object]:
        values = asdict(self)
        for name in (
            "blur_score",
            "brightness",
            "contrast",
            "skew_angle",
            "skew_confidence",
            "orientation_confidence",
            "perspective_confidence",
        ):
            values[name] = f"{values[name]:.3f}"
        for name in (
            "orientation_applied",
            "perspective_corrected",
            "crop_applied",
            "deskew_applied",
            "contrast_enhanced",
        ):
            values[name] = "yes" if values[name] else "no"
        return values


def _rotate_right_angle(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Unsupported right-angle rotation: {angle}")


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _detect_document_quad(image: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Find a dominant paper quadrilateral and score its reliability.

    Edge contours work for clean scans. A second brightness-mask pass handles
    phone photos with soft paper edges, black letterboxing, or a white PDF
    canvas around the embedded photograph.
    """

    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    small = image
    if scale < 1.0:
        small = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edge_contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    _, bright_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask_kernel_size = max(5, round(min(gray.shape) * 0.012))
    if mask_kernel_size % 2 == 0:
        mask_kernel_size += 1
    mask_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (mask_kernel_size, mask_kernel_size)
    )
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_CLOSE, mask_kernel, iterations=2
    )
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_OPEN, mask_kernel, iterations=1
    )
    bright_contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = small.shape[0] * small.shape[1]
    best_quad: np.ndarray | None = None
    best_confidence = 0.0
    contours = sorted(
        [*edge_contours, *bright_contours],
        key=cv2.contourArea,
        reverse=True,
    )[:20]
    for contour in contours:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        quad = None
        for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.05, 0.07):
            candidate = cv2.approxPolyDP(
                hull, epsilon_ratio * perimeter, True
            )
            if len(candidate) == 4 and cv2.isContourConvex(candidate):
                quad = candidate
                break
        if quad is None:
            continue
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / image_area
        if not MIN_DOCUMENT_AREA_RATIO <= area_ratio <= MAX_DOCUMENT_AREA_RATIO:
            continue
        quad_area = cv2.contourArea(quad)
        rectangularity = contour_area / max(quad_area, 1.0)
        ordered = _order_quad(quad.reshape(4, 2))
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(mask, ordered.astype(np.int32), 255)
        inside_mean = float(cv2.mean(gray, mask=mask)[0])
        outside_mask = cv2.bitwise_not(mask)
        outside_ratio = cv2.countNonZero(outside_mask) / image_area
        if outside_ratio < 0.03:
            continue
        outside_mean = float(cv2.mean(gray, mask=outside_mask)[0])
        background_separation = inside_mean - outside_mean
        # A paper boundary must separate a lighter document from its
        # surroundings. This prevents large invoice tables from being cropped.
        if background_separation < 12.0:
            continue
        top, right, bottom, left = (
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
            np.linalg.norm(ordered[2] - ordered[3]),
            np.linalg.norm(ordered[3] - ordered[0]),
        )
        opposite_similarity = min(top, bottom) / max(top, bottom, 1.0)
        opposite_similarity *= min(left, right) / max(left, right, 1.0)
        background_score = min(1.0, background_separation / 80.0)
        confidence = (
            min(1.0, area_ratio / 0.45) * 0.25
            + min(1.0, rectangularity) * 0.25
            + opposite_similarity * 0.15
            + background_score * 0.35
        )
        if confidence > best_confidence:
            best_quad = ordered / scale
            best_confidence = confidence
    return best_quad, float(min(1.0, best_confidence))


def _warp_document(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = quad
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    target_width = max(1, round(float(width)))
    target_height = max(1, round(float(height)))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _analysis_gray(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(1.0, MAX_ANALYSIS_DIMENSION / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return gray, scale


def _estimate_skew(gray: np.ndarray) -> tuple[float, float]:
    """Estimate a small page skew with a bounded horizontal Hough search."""

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Restrict theta to near-horizontal lines. Unlike HoughLinesP, this avoids
    # an unbounded segment search on dense, high-resolution invoice pages.
    lines = cv2.HoughLines(
        edges,
        1,
        np.pi / 720,
        threshold=max(30, round(gray.shape[1] * 0.04)),
        min_theta=np.deg2rad(80),
        max_theta=np.deg2rad(100),
    )
    if lines is None:
        return 0.0, 0.0

    angles = [
        math.degrees(float(theta)) - 90.0
        for _, theta in np.asarray(lines).reshape(-1, 2)[:200]
    ]
    if len(angles) < 3:
        return 0.0, 0.0

    median = float(np.median(angles))
    agreement = sum(abs(angle - median) <= 1.0 for angle in angles) / len(angles)
    evidence = min(1.0, len(angles) / 12.0)
    confidence = min(1.0, agreement * 0.75 + evidence * 0.25)
    return float(median), float(confidence)


def _rotate_without_cropping(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(math.ceil(height * sine + width * cosine))
    new_height = int(math.ceil(height * cosine + width * sine))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _enhance_low_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )


def normalize_image(
    path: str | Path,
    *,
    orientation_classifier: Callable[[str | Path], tuple[int, float]] | None = None,
) -> ImageQualityResult:
    """Normalize one image in place and return auditable quality metadata.

    The operation is intentionally conservative: RGB/PNG normalization is
    unconditional, while deskew and contrast enhancement require strong
    evidence. Ambiguous pages are flagged instead of modified aggressively.
    """

    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV cannot read image: {image_path}")

    flags: list[str] = []
    review_flags: list[str] = []

    orientation_angle = 0
    orientation_confidence = 0.0
    orientation_applied = False
    if orientation_classifier is not None:
        orientation_angle, orientation_confidence = orientation_classifier(image_path)
        if orientation_angle and orientation_confidence >= MIN_ORIENTATION_CONFIDENCE:
            image = _rotate_right_angle(image, orientation_angle)
            orientation_applied = True
            flags.append(f"rotated_{orientation_angle}")
        elif orientation_angle:
            review_flags.append("orientation_uncertain")

    document_quad, perspective_confidence = _detect_document_quad(image)
    perspective_corrected = (
        document_quad is not None
        and perspective_confidence >= MIN_PERSPECTIVE_CONFIDENCE
    )
    crop_applied = perspective_corrected
    if perspective_corrected:
        image = _warp_document(image, document_quad)
        flags.extend(("perspective_corrected", "cropped"))
    elif document_quad is not None:
        review_flags.append("document_contour_uncertain")

    gray, _ = _analysis_gray(image)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    skew_angle, skew_confidence = _estimate_skew(gray)

    if blur_score < BLUR_REVIEW_THRESHOLD:
        review_flags.append("blurry")
    if brightness < DARK_REVIEW_THRESHOLD:
        review_flags.append("dark")
    elif brightness > BRIGHT_REVIEW_THRESHOLD:
        review_flags.append("overexposed")

    deskew_applied = (
        MIN_DESKEW_ANGLE <= abs(skew_angle) <= MAX_DESKEW_ANGLE
        and skew_confidence >= MIN_DESKEW_CONFIDENCE
    )
    if deskew_applied:
        # If content slopes by +angle, rotate it by -angle to make it horizontal.
        image = _rotate_without_cropping(image, -skew_angle)
        flags.append("deskewed")
    elif abs(skew_angle) > MAX_DESKEW_ANGLE:
        review_flags.append("large_skew")
    elif (
        abs(skew_angle) >= MIN_DESKEW_ANGLE
        and skew_confidence < MIN_DESKEW_CONFIDENCE
    ):
        review_flags.append("skew_uncertain")

    contrast_enhanced = contrast < LOW_CONTRAST_THRESHOLD
    if contrast_enhanced:
        image = _enhance_low_contrast(image)
        flags.append("contrast_enhanced")

    temporary_path = image_path.with_name(f".{image_path.stem}.normalized.png")
    if not cv2.imwrite(str(temporary_path), image):
        raise ValueError(f"OpenCV cannot write normalized image: {temporary_path}")
    temporary_path.replace(image_path)

    all_flags = flags + review_flags
    if review_flags:
        status = "review"
    elif flags:
        status = "corrected"
    else:
        status = "accepted"
    return ImageQualityResult(
        orientation_angle=orientation_angle,
        orientation_confidence=orientation_confidence,
        orientation_applied=orientation_applied,
        perspective_confidence=perspective_confidence,
        perspective_corrected=perspective_corrected,
        crop_applied=crop_applied,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        skew_angle=skew_angle,
        skew_confidence=skew_confidence,
        deskew_applied=deskew_applied,
        contrast_enhanced=contrast_enhanced,
        quality_status=status,
        quality_flags=";".join(all_flags),
    )
