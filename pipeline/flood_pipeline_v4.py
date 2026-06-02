# ============================================================
# pipeline/flood_pipeline_v4.py
# Integrated Pipeline - v4 3-class classifier + segmentation
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from classification.inference.classifier_engine_v4 import FloodClassifierEngineV4
from segmentation.advanced_segmenter import AdvancedFloodSegmenter


class FloodPipelineV4:
    def __init__(self):
        self.classifier = FloodClassifierEngineV4()
        self.segmenter = AdvancedFloodSegmenter()

    def run(self, image):
        classification = self.classifier.predict(image)
        classification["flood_probability"] = classification["probabilities"].get(
            "real_flood", 0.0
        )

        if not classification["run_segmentation"]:
            decision = (
                "NON_FLOOD_IMAGE"
                if classification["decision"] == "MAP_DIAGRAM"
                else "NO FLOOD"
            )
            return {
                "decision": decision,
                "segmentation": None,
                "classification": classification,
            }

        segmentation = self.segmenter.segment(image)

        return {
            "decision": "FLOOD",
            "classification": classification,
            "segmentation": segmentation,
        }
