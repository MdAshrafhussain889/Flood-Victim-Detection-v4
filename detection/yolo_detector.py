# ============================================================
# detection/yolo_detector.py
# YOLOv8 person detection (with low-quality image handling)
# ============================================================

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_IMG_SIZE,
    YOLO_UPSCALE_MIN_DIM,
    PERSON_CLASS_ID,
)


def _enhance_low_quality(image):
    """Improve contrast for dark / blurry flood scenes."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)


def _prepare_for_yolo(image):
    """
    Upscale small images and enhance contrast so YOLO can find people.
    Returns (processed_image, scale) where scale maps proc coords → original.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    out = image

    if max_dim < YOLO_UPSCALE_MIN_DIM:
        scale = YOLO_UPSCALE_MIN_DIM / max_dim
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if max_dim < 900 or min(h, w) < 200:
        out = _enhance_low_quality(out)

    return out, scale


class YOLOPersonDetector:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(YOLO_MODEL)

    def detect(self, image):
        proc, scale = _prepare_for_yolo(image)
        inv = 1.0 / scale

        results = self.model(
            proc,
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMG_SIZE,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1 * inv)
                y1 = int(y1 * inv)
                x2 = int(x2 * inv)
                y2 = int(y2 * inv)
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0]),
                    }
                )
        return detections
