# ============================================================
# classification/inference/classifier_engine_v4.py
# 3-class flood classifier inference engine (v4)
# ============================================================

import cv2
import torch
import numpy as np
import sys
import os
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from classification.models.efficientnet_classifier_v4 import FloodClassifierV4
from classification.metrics_v4 import CLASS_NAMES
from configs.config import CLASSIFIER_IMG_SIZE, CLASSIFIER_V4_CHECKPOINT

CHECKPOINT_PATH = CLASSIFIER_V4_CHECKPOINT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_ID_TO_DECISION = {
    0: "NO_FLOOD",
    1: "FLOOD",
    2: "MAP_DIAGRAM",
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FloodClassifierEngineV4:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"v4 classifier checkpoint not found: {checkpoint_path}. "
                "Run classification/training/train_classifier_v4.py first."
            )
        self.model = FloodClassifierV4(pretrained=False)
        state_dict = self._load_checkpoint(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(DEVICE).eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=DEVICE, weights_only=True
            )
        except TypeError:
            warnings.warn(
                "PyTorch version does not support weights_only=True; using legacy loader.",
                RuntimeWarning,
            )
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict) or not checkpoint:
            raise RuntimeError(f"Invalid checkpoint format: {checkpoint_path}")
        return checkpoint

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(DEVICE)

    @torch.no_grad()
    def predict(self, image):
        tensor = self.preprocess(image)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_id = int(np.argmax(probs))
        class_name = CLASS_ID_TO_NAME[class_id]
        confidence = float(probs[class_id])
        decision = CLASS_ID_TO_DECISION[class_id]
        run_segmentation = class_id == 1

        probabilities = {
            CLASS_ID_TO_NAME[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        return {
            "decision": decision,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "probabilities": probabilities,
            "run_segmentation": run_segmentation,
        }
