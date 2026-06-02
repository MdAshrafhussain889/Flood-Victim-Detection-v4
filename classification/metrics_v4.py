# ============================================================
# classification/metrics_v4.py
# 3-class classification metrics (v4)
# ============================================================

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

CLASS_NAMES = ["non_flood", "real_flood", "maps_diagrams"]
NUM_CLASSES = 3


def compute_metrics_v4(logits, targets):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = targets.long().cpu().numpy()

    macro_precision = precision_score(
        targets, preds, average="macro", zero_division=0, labels=list(range(NUM_CLASSES))
    )
    macro_recall = recall_score(
        targets, preds, average="macro", zero_division=0, labels=list(range(NUM_CLASSES))
    )
    macro_f1 = f1_score(
        targets, preds, average="macro", zero_division=0, labels=list(range(NUM_CLASSES))
    )

    per_class_precision = precision_score(
        targets, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES))
    )
    per_class_recall = recall_score(
        targets, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES))
    )
    per_class_f1 = f1_score(
        targets, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES))
    )

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
        }

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }
