# ============================================================
# classification/dataset_v4.py
# 3-class flood classification dataset loader (v4)
# ============================================================

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class FloodClassificationDatasetV4(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {row.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row.label)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
