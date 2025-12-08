#!/usr/bin/env python3
"""
vmae_infer_v.py
Hard-coded inference script:
 - Uses splits_vaibhav.json
 - Uses checkpoint: best_vmae_16f224_updated_binary.pth
"""

import os
import json
from pathlib import Path
import numpy as np
import csv
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import VideoMAEForVideoClassification

# ---------------------------
# HARDCODED CONFIG
# ---------------------------
SPLIT_FILE = "splits_vaibhav.json"
CHECKPOINT = "best_vmae_16f224_updated_binary.pth"
BATCH_SIZE = 8
NUM_WORKERS = 4
OUT_CSV = "predictions_test.csv"

MODEL_NAME = "MCG-NJU/videomae-base"
IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 16

# ---------------------------
# Transforms
# ---------------------------
frame_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_video_frames(video_path, num_frames=FRAMES_PER_VIDEO, image_size=IMAGE_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return torch.zeros(num_frames, 3, image_size, image_size)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            f = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (image_size, image_size))
        frames.append(f)

    cap.release()

    try:
        frames_t = torch.stack([frame_transform(f) for f in frames])
    except:
        frames_t = torch.zeros(num_frames, 3, image_size, image_size)

    return frames_t

# ---------------------------
# Dataset
# ---------------------------
class VideoFolderDataset(Dataset):
    def __init__(self, split_entries):
        self.samples = split_entries

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bin_label, fine_idx = self.samples[idx]
        frames = load_video_frames(path)
        return frames, torch.tensor(fine_idx), torch.tensor(bin_label), path

def collate_videos(batch):
    frames = torch.stack([b[0] for b in batch], dim=0)
    fine = torch.stack([b[1] for b in batch], dim=0)
    bin_ = torch.stack([b[2] for b in batch], dim=0)
    paths = [b[3] for b in batch]
    return frames, fine, bin_, paths

# ---------------------------
# Model
# ---------------------------
class VideoMAEWithBinary(nn.Module):
    def __init__(self, num_fine):
        super().__init__()
        self.backbone = VideoMAEForVideoClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_fine,
            ignore_mismatched_sizes=True
        )
        self.binary_head = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, pixel_values, output_hidden_states=True):
        out = self.backbone(pixel_values=pixel_values,
                            output_hidden_states=output_hidden_states)
        logits_fine = out.logits
        cls_tokens = out.hidden_states[-1][:, 0, :]
        logits_bin = self.binary_head(cls_tokens)
        return logits_fine, logits_bin

# ---------------------------
# Checkpoint loader
# ---------------------------
def load_checkpoint_safe(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    fixed = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        fixed[k] = v

    model.load_state_dict(fixed, strict=False)
    print("Loaded checkpoint:", path)

# ---------------------------
# Evaluation loop
# ---------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    total = 0
    correct_f = 0
    correct_b = 0

    with torch.no_grad():
        for videos, fine_labels, bin_labels, paths in tqdm(dataloader):
            videos = videos.to(device)
            fine_labels = fine_labels.to(device)
            bin_labels = bin_labels.to(device)

            logits_f, logits_b = model(videos)
            pf = logits_f.argmax(1)
            pb = logits_b.argmax(1)

            correct_f += (pf == fine_labels).sum().item()
            correct_b += (pb == bin_labels).sum().item()
            total += fine_labels.size(0)

            for p, tf, tb, pp, pb_ in zip(paths, fine_labels.cpu(), bin_labels.cpu(),
                                          pf.cpu(), pb.cpu()):
                preds.append({
                    "path": p,
                    "true_fine": int(tf),
                    "true_bin": int(tb),
                    "pred_fine": int(pp),
                    "pred_bin": int(pb_),
                })

    return correct_f / total, correct_b / total, preds

# ---------------------------
# MAIN (no argparse)
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load split file
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(SPLIT_FILE)

    with open(SPLIT_FILE, "r") as f:
        data = json.load(f)

    # pick test split "val"
    test_entries = data["val"]

    # compute number of fine classes
    num_fine = max([e[2] for e in test_entries]) + 1

    print(f"Test samples: {len(test_entries)}, fine classes: {num_fine}")

    # dataset + loader
    test_ds = VideoFolderDataset(test_entries)
    test_loader = DataLoader(test_ds,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             collate_fn=collate_videos,
                             num_workers=NUM_WORKERS)

    # model
    model = VideoMAEWithBinary(num_fine).to(device)

    # load checkpoint
    load_checkpoint_safe(CHECKPOINT, model, device)

    # evaluate
    acc_f, acc_b, preds = evaluate(model, test_loader, device)

    print(f"\nFine Acc: {acc_f*100:.2f}%, Binary Acc: {acc_b*100:.2f}%")

    # save CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "true_fine", "true_bin", "pred_fine", "pred_bin"])
        w.writeheader()
        for x in preds:
            w.writerow(x)

    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
