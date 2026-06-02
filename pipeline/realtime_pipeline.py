# ============================================================
# pipeline/realtime_pipeline.py
# Real-Time Flood System - v4 classifier + detection
# ============================================================

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.flood_pipeline_v4 import FloodPipelineV4
from classification.inference.classifier_engine_v4 import FloodClassifierEngineV4
from detection.yolo_detector import YOLOPersonDetector
from tracking.simple_tracker import SimpleTracker


class RealTimeFloodSystem:
    def __init__(self):
        self.pipeline = FloodPipelineV4()
        if not isinstance(self.pipeline.classifier, FloodClassifierEngineV4):
            raise RuntimeError(
                "Expected FloodClassifierEngineV4 in pipeline; "
                "restart Streamlit (cached old v3 models) or check pipeline imports."
            )
        self.detector = YOLOPersonDetector()
        self.tracker  = SimpleTracker()

    def process_frame(self, image):
        result = self.pipeline.run(image)

        if result["decision"] != "FLOOD":
            result["detections"] = []
            return result

        detections = self.detector.detect(image)
        detections = self.tracker.update(detections)
        for det in detections:
            det.pop("risk", None)
            det.pop("risk_score", None)
            det.pop("overlap", None)
            det.pop("water_contact", None)
        result["detections"] = detections
        return result
