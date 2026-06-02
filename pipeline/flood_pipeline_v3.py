# ============================================================
# pipeline/flood_pipeline_v3.py
# Backward-compatible alias — always delegates to v4 pipeline.
# ============================================================

from pipeline.flood_pipeline_v4 import FloodPipelineV4


class FloodPipelineV3(FloodPipelineV4):
    """Deprecated name kept for imports; runs the v4 3-class classifier."""
