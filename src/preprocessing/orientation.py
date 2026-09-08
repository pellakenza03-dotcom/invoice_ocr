"""Persistent PaddleOCR document-orientation classifier."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Callable

import cv2


class DocumentOrientationService:
    """Load PP-LCNet once and classify page rotations sequentially on CPU."""

    def __init__(
        self,
        *,
        model_factory: Callable[..., Any] | None = None,
        cpu_threads: int = 4,
    ) -> None:
        if model_factory is None:
            from paddleocr import DocImgOrientationClassification

            model_factory = DocImgOrientationClassification
        self._model = model_factory(
            model_name="PP-LCNet_x1_0_doc_ori",
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=cpu_threads,
            topk=4,
        )
        self._lock = Lock()

    def predict(self, image_path: str | Path) -> tuple[int, float]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image for orientation: {image_path}")
        candidate_angles = (0, 90, 180, 270)
        candidates = [
            image,
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(image, cv2.ROTATE_180),
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        ]
        with self._lock:
            results = list(self._model.predict(candidates, batch_size=4))
        if len(results) != 4:
            raise ValueError(
                f"Expected four orientation results, received {len(results)}."
            )

        upright_scores: list[float] = []
        for result in results:
            labels = list(result.get("label_names") or ())
            scores = result.get("scores")
            if scores is None or len(scores) != len(labels) or "0" not in labels:
                raise ValueError(
                    "Orientation result does not contain all class probabilities."
                )
            upright_scores.append(float(scores[labels.index("0")]))

        best_index = max(range(4), key=upright_scores.__getitem__)
        return candidate_angles[best_index], upright_scores[best_index]
