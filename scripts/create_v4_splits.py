from pathlib import Path
import csv
import random
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_v4"
OUT_DIR = ROOT / "splits_v4"
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = [
    (DATA_DIR / "non_flood", 0, "non_flood"),
    (DATA_DIR / "hard_negatives", 0, "non_flood"),
    (DATA_DIR / "real_flood", 1, "real_flood"),
    (DATA_DIR / "maps_diagrams", 2, "maps_diagrams"),
]

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def collect_rows():
    rows = []
    for folder, label, class_name in SOURCES:
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not files:
            raise RuntimeError(f"No image files found in: {folder}")

        for path in files:
            rows.append({
                "image_path": str(path),
                "label": label,
                "class_name": class_name,
                "source_folder": folder.name,
            })
    return rows


def stratified_split(rows):
    rng = random.Random(SEED)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    splits = {"train": [], "val": [], "test": []}

    for label, label_rows in by_label.items():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits["train"].extend(label_rows[:n_train])
        splits["val"].extend(label_rows[n_train:n_train + n_val])
        splits["test"].extend(label_rows[n_train + n_val:])

    for split_rows in splits.values():
        rng.shuffle(split_rows)

    return splits


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "class_name", "source_folder"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(name, rows):
    label_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(row["source_folder"] for row in rows)
    print(f"\n{name}: {len(rows)} images")
    print("  labels:", dict(sorted(label_counts.items())))
    print("  sources:", dict(sorted(source_counts.items())))


def main():
    rows = collect_rows()
    splits = stratified_split(rows)

    for name, split_rows in splits.items():
        write_csv(OUT_DIR / f"{name}.csv", split_rows)

    print("Created v4 split files in:", OUT_DIR)
    print_summary("ALL", rows)
    for name in ["train", "val", "test"]:
        print_summary(name.upper(), splits[name])


if __name__ == "__main__":
    main()
