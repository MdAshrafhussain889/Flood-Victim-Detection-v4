# ============================================================
# classification/training/evaluate_classifier_v4.py
# 3-class classifier evaluation on test split (v4)
# ============================================================

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from classification.dataset_v4 import FloodClassificationDatasetV4
from classification.transforms import get_val_transform
from classification.models.efficientnet_classifier_v4 import FloodClassifierV4
from classification.metrics_v4 import compute_metrics_v4, CLASS_NAMES

TEST_CSV = os.path.join(ROOT, "splits_v4", "test.csv")
BEST_CHECKPOINT = os.path.join(ROOT, "checkpoints_v4", "flood_classifier_v4_best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cuda" else 16


def main():
    if not os.path.exists(BEST_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {BEST_CHECKPOINT}. "
            "Run train_classifier_v4.py first."
        )

    test_dataset = FloodClassificationDatasetV4(TEST_CSV, transform=get_val_transform())
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = FloodClassifierV4(pretrained=False).to(DEVICE)
    try:
        state = torch.load(BEST_CHECKPOINT, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(BEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(images)
            all_logits.append(logits)
            all_targets.append(labels)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets_np = targets.long().cpu().numpy()

    metrics = compute_metrics_v4(logits, targets)

    print("=" * 60)
    print("Test set evaluation (v4)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {BEST_CHECKPOINT}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")

    print("\nPer-class metrics:")
    for class_name, class_metrics in metrics["per_class"].items():
        print(
            f"  {class_name}: P={class_metrics['precision']:.4f} "
            f"R={class_metrics['recall']:.4f} F1={class_metrics['f1']:.4f}"
        )

    print("\nClassification report:")
    print(
        classification_report(
            targets_np,
            preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(targets_np, preds, labels=list(range(len(CLASS_NAMES))))
    print(f"{'':16s}", " ".join(f"{name:>14s}" for name in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:16s}", " ".join(f"{cm[i, j]:14d}" for j in range(len(CLASS_NAMES))))


if __name__ == "__main__":
    main()
