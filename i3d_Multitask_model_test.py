#!/usr/bin/env python3
"""
i3d_test.py

Runs evaluation on the test split using the same preprocessing used in training.
"""

import os
import json
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import video
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# CONFIG
# -----------------------------
NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = "splits.json"
BEST_CKPT = "checkpoints/best_i3d_multitask.pth"
FINAL_CKPT = "final_i3d_multitask.pth"
OUT_CSV = "test_predictions.csv"

# -----------------------------
# NORMALIZATION (same as training)
# -----------------------------
transform_frame = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # SAME AS TRAINING
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# UNIFORM FRAME SAMPLING
# -----------------------------
def sample_uniform(frames, num_frames):
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [frames[i] for i in idx]


def load_video_frames(video_path, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE):
    """Load video → return tensor [T, C, H, W]"""
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception:
        return torch.zeros(num_frames, 3, image_size, image_size)

    if not cap.isOpened():
        return torch.zeros(num_frames, 3, image_size, image_size)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return torch.zeros(num_frames, 3, image_size, image_size)

    sampled = sample_uniform(frames, num_frames)
    tensors = [transform_frame(f) for f in sampled]
    return torch.stack(tensors)


# -----------------------------
# DATASET
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bin_label, fine_label = self.samples[idx]
        vid = load_video_frames(path)
        return vid, int(bin_label), int(fine_label), path


# -----------------------------
# MODEL (same as training)
# -----------------------------
class MultiTaskI3D(nn.Module):
    def __init__(self, num_fine):
        super().__init__()
        self.backbone = video.mc3_18(weights="DEFAULT")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.binary_head = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.fine_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(),
            nn.Linear(512, num_fine)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        feats = self.backbone(x)
        return self.binary_head(feats), self.fine_head(feats)


# -----------------------------
# METRIC UTILS (confusion matrix)
# -----------------------------
def confusion_matrix(true, pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def compute_metrics(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall,
                   precision + recall,
                   out=np.zeros_like(tp, float),
                   where=(precision + recall) != 0)

    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1))
    }
    return macro


# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():

    # --- load splits ---
    if not os.path.exists(SPLITS):
        raise FileNotFoundError("splits.json not found.")

    splits = json.load(open(SPLITS))
    test_samples = splits["test"]
    print("Test samples:", len(test_samples))

    # Extract num_fine classes
    all_fine = [s[2] for s in splits["train"]] + [s[2] for s in splits["val"]] + [s[2] for s in splits["test"]]
    num_fine = max(all_fine) + 1

    # dataset + loader
    test_ds = VideoDataset(test_samples)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # model
    model = MultiTaskI3D(num_fine=num_fine).to(DEVICE)

    # checkpoint load
    ckpt = BEST_CKPT if os.path.exists(BEST_CKPT) else FINAL_CKPT
    print("Loading checkpoint:", ckpt)

    ckpt_data = torch.load(ckpt, map_location='cpu')

    # Fix: Automatically detect training checkpoint vs pure state_dict
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        print("Loaded full training checkpoint.")
        state_dict = ckpt_data["model_state"]
    else:
        print("Loaded raw state_dict.")
        state_dict = ckpt_data

    model.load_state_dict(state_dict)


    model.eval()

    all_bin_true = []
    all_bin_pred = []
    all_fine_true = []
    all_fine_pred = []

    softmax = nn.Softmax(dim=1)

    rows = []

    with torch.no_grad():
        for vids, bins, fines, paths in tqdm(test_loader, desc="Testing"):
            vids = vids.to(DEVICE).float()

            out_b, out_f = model(vids)
            pb = softmax(out_b).cpu().numpy()
            pf = softmax(out_f).cpu().numpy()

            pred_b = pb.argmax(axis=1)
            pred_f = pf.argmax(axis=1)

            for i in range(len(paths)):
                all_bin_true.append(int(bins[i]))
                all_bin_pred.append(int(pred_b[i]))
                all_fine_true.append(int(fines[i]))
                all_fine_pred.append(int(pred_f[i]))

                rows.append([
                    paths[i], int(bins[i]), int(pred_b[i]), float(pb[i, pred_b[i]]),
                    int(fines[i]), int(pred_f[i]), float(pf[i, pred_f[i]])
                ])

    # save CSV
    header = ["video", "bin_true", "bin_pred", "bin_conf", "fine_true", "fine_pred", "fine_conf"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("Saved predictions to", OUT_CSV)

    # accuracy
    bin_acc = np.mean(np.array(all_bin_true) == np.array(all_bin_pred)) * 100
    fine_acc = np.mean(np.array(all_fine_true) == np.array(all_fine_pred)) * 100

    print(f"\nBinary Accuracy: {bin_acc:.2f}%")
    print(f"Fine Accuracy:   {fine_acc:.2f}%")

    # confusion matrices
    bin_cm = confusion_matrix(all_bin_true, all_bin_pred, 2)
    fine_cm = confusion_matrix(all_fine_true, all_fine_pred, num_fine)

    # metrics
    bin_metrics = compute_metrics(bin_cm)
    fine_metrics = compute_metrics(fine_cm)

    print("\nBinary metrics:", bin_metrics)
    print("Fine metrics:", fine_metrics)

    json.dump({
        "binary_accuracy": bin_acc,
        "fine_accuracy": fine_acc,
        "binary_cm": bin_cm.tolist(),
        "fine_cm_shape": fine_cm.shape,
        "binary_metrics": bin_metrics,
        "fine_metrics": fine_metrics
    }, open("test_summary.json", "w"), indent=2)

    print("Saved test_summary.json")


if __name__ == "__main__":
    main()

