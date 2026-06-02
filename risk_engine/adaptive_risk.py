# ============================================================
# risk_engine/adaptive_risk.py
# Per-person risk from flood-mask overlap (primary signal)
# ============================================================

import numpy as np

# Water-contact bands (fraction 0–1) → risk label
RISK_CRITICAL_WATER = 0.55
RISK_HIGH_WATER = 0.35
RISK_MEDIUM_WATER = 0.18


def compute_body_visibility(box):
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    area = width * height
    return min(area / 50000.0, 1.0)


def compute_local_water_density(mask, box):
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.mean(region > 0))


def compute_overlap(mask, box):
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.mean(region > 0))


def compute_water_contact(overlap, density):
    """How much of the person is in flooded area (main risk driver)."""
    return 0.65 * overlap + 0.35 * density


def compute_risk_score(overlap, density, confidence, visibility):
    """0–1 score for sorting; classification uses water_contact bands."""
    water = compute_water_contact(overlap, density)
    return float(min(1.0, water))


def classify_risk(overlap, density, confidence, visibility):
    """
    Risk level from water contact with the flood mask inside the person box.
    CRITICAL / HIGH / MEDIUM / LOW — not related to YOLO or classifier confidence.
    """
    water = compute_water_contact(overlap, density)
    if water >= RISK_CRITICAL_WATER:
        return "CRITICAL"
    if water >= RISK_HIGH_WATER:
        return "HIGH"
    if water >= RISK_MEDIUM_WATER:
        return "MEDIUM"
    return "LOW"


def risk_explanation(overlap_pct, density_pct, water_contact_pct, risk):
    return (
        f"{risk}: ~{water_contact_pct:.0f}% of this person overlaps flood water "
        f"(mask overlap {overlap_pct:.0f}%, local density {density_pct:.0f}%)."
    )
