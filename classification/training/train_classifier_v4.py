# ============================================================
# classification/training/train_classifier_v4.py
# 3-class classifier training pipeline (v4)
# ============================================================

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from classification.dataset_v4 import FloodClassificationDatasetV4
from classification.transforms import get_train_transform, get_val_transform
from classification.models.efficientnet_classifier_v4 import FloodClassifierV4
from classification.metrics_v4 import compute_metrics_v4, CLASS_NAMES

TRAIN_CSV = os.path.join(ROOT, "splits_v4", "train.csv")
VAL_CSV = os.path.join(ROOT, "splits_v4", "val.csv")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints_v4")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "flood_classifier_v4_best.pth")
LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "flood_classifier_v4_last.pth")
V3_CHECKPOINT = os.path.join(ROOT, "checkpoints", "flood_classifier_best.pth")

EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cuda" else 16


def compute_class_weights(csv_path, device):
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().sort_index()
    n_samples = len(df)
    n_classes = len(counts)
    weights = n_samples / (n_classes * counts.astype(float))
    weight_tensor = torch.tensor(
        [weights[i] for i in range(len(counts))],
        dtype=torch.float32,
        device=device,
    )
    print("Class weights:", {CLASS_NAMES[i]: round(weight_tensor[i].item(), 4) for i in range(len(counts))})
    return weight_tensor


def load_v3_backbone_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"No v3 checkpoint at {checkpoint_path}; using ImageNet pretrained backbone.")
        return

    try:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for key, value in state.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                skipped.append(key)
                continue
            filtered[key] = value

        model.load_state_dict(filtered, strict=False)
        print(
            f"Loaded {len(filtered)} v3 weights into v4 model "
            f"(skipped {len(skipped)} mismatched keys, e.g. old 1-class head)."
        )
    except Exception as exc:
        print(f"Warning: failed to load v3 weights ({exc}); continuing with ImageNet init.")


def print_metrics(prefix, metrics):
    print(f"{prefix} accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix} macro precision: {metrics['macro_precision']:.4f}")
    print(f"{prefix} macro recall: {metrics['macro_recall']:.4f}")
    print(f"{prefix} macro F1: {metrics['macro_f1']:.4f}")
    for class_name, class_metrics in metrics["per_class"].items():
        print(
            f"{prefix} {class_name}: "
            f"P={class_metrics['precision']:.4f} "
            f"R={class_metrics['recall']:.4f} "
            f"F1={class_metrics['f1']:.4f}"
        )


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    all_logits = []
    all_targets = []
    running_loss = 0.0
    bar = tqdm(loader, leave=False)

    for batch in bar:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        all_logits.append(logits.detach())
        all_targets.append(labels.detach())
        bar.set_description(f"{'Train' if is_train else 'Val'} Loss: {loss.item():.4f}")

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = compute_metrics_v4(logits, targets)
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


def main():
    print(f"Device: {DEVICE}, batch size: {BATCH_SIZE}")

    train_dataset = FloodClassificationDatasetV4(TRAIN_CSV, transform=get_train_transform())
    val_dataset = FloodClassificationDatasetV4(VAL_CSV, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = FloodClassifierV4(pretrained=True).to(DEVICE)
    load_v3_backbone_weights(model, V3_CHECKPOINT)

    class_weights = compute_class_weights(TRAIN_CSV, DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion)

        print(f"\nTrain loss: {train_metrics['loss']:.4f}")
        print_metrics("Train", train_metrics)
        print(f"\nVal loss: {val_metrics['loss']:.4f}")
        print_metrics("Val", val_metrics)

        torch.save(model.state_dict(), LAST_CHECKPOINT)
        print(f"\nLast checkpoint saved to {LAST_CHECKPOINT}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), BEST_CHECKPOINT)
            print(f"Best model saved to {BEST_CHECKPOINT} (val macro F1: {best_val_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best val macro F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
