#!/usr/bin/env python3
# vmae_train.py
# VideoMAE fine-tuning with fine-grained + binary heads, AMP, resume, ReduceLROnPlateau,
# Data-parallel support, saved splits (splits_vaibhav.json), long-run (epochs=200) + patience=20.

import os
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import VideoMAEForVideoClassification
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------
# CONFIG (adjust paths/hyperparams here)
# ---------------------------
DATA_ROOT = "UCF_MERGED_NORM"
FRAMES_PER_VIDEO = 16        # VideoMAE stable setting (change if you know the checkpoint supports different)
IMAGE_SIZE = 224
BATCH_SIZE = 8               # you requested BATCH_SIZE = 8
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 4
PIN_MEMORY = True
BENCHMARK = True
PREFETCH_FACTOR = 2

EPOCHS = 200                 # total epochs to run
PATIENCE = 20                # early stop patience (you asked for 20)
LR = 1e-4
WEIGHT_DECAY = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the usual public base; if you want a different checkpoint change this.
MODEL_NAME = "MCG-NJU/videomae-base"

CHECKPOINT_PATH = "best_vmae_16f224_updated_binary.pth"
SPLIT_FILE = "splits_vaibhav.json"

# Scheduler params (ReduceLROnPlateau)
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 10
SCHED_MIN_LR = 1e-6

# ---------------------------
# Sanity / backend
# ---------------------------
torch.backends.cudnn.benchmark = BENCHMARK

# ---------------------------
# Transforms
# ---------------------------
frame_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# Video loader: returns (T, C, H, W)
# ---------------------------
def load_video_frames(video_path, num_frames=FRAMES_PER_VIDEO, image_size=IMAGE_SIZE):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            f = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (image_size, image_size))
        frames.append(f)
    cap.release()

    try:
        frames_t = torch.stack([frame_transform(f) for f in frames])  # (T, C, H, W)
    except Exception:
        frames_t = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)

    return frames_t

# ---------------------------
# Dataset (expects split entries: [abs_path, bin_label, fine_idx])
# ---------------------------
class VideoFolderDataset(Dataset):
    def __init__(self, split_entries):
        self.samples = split_entries

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bin_label, fine_idx = self.samples[idx]
        frames = load_video_frames(path)
        # model expects shape (B, T, C, H, W) later; here we return (T,C,H,W)
        return frames, torch.tensor(fine_idx, dtype=torch.long), torch.tensor(bin_label, dtype=torch.long)

def collate_videos(batch):
    # batch: list of (frames[T,C,H,W], fine_label, bin_label)
    frames = torch.stack([item[0] for item in batch], dim=0)   # (B, T, C, H, W)
    fine = torch.stack([item[1] for item in batch], dim=0)
    bin_ = torch.stack([item[2] for item in batch], dim=0)
    return frames, fine, bin_

# ---------------------------
# Build sample list from folder structure
# ---------------------------
def build_samples(root_dir):
    root = Path(root_dir)
    samples = []           # (abs_path, bin_label, fine_name)
    fine_classes = set()

    for top in ["Normal", "Abnormal"]:
        top_dir = root / top
        if not top_dir.exists():
            continue
        for cls in sorted(x for x in top_dir.iterdir() if x.is_dir()):
            fine_classes.add(cls.name)
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                for p in cls.glob(ext):
                    bin_label = 0 if top == "Normal" else 1
                    samples.append((str(p.resolve()), bin_label, cls.name))

    fine_classes = sorted(list(fine_classes))
    fine_to_idx = {c: i for i, c in enumerate(fine_classes)}

    final = [
        [path, bin_label, fine_to_idx[fine_name]]
        for (path, bin_label, fine_name) in samples
    ]
    return final

# ---------------------------
# Model wrapper with binary head
# ---------------------------
class VideoMAEWithBinary(nn.Module):
    def __init__(self, num_fine_classes):
        super().__init__()
        try:
            self.backbone = VideoMAEForVideoClassification.from_pretrained(
                MODEL_NAME,
                num_labels=num_fine_classes,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            raise RuntimeError(f"Error loading model '{MODEL_NAME}': {e}")
        self.binary_head = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, pixel_values, output_hidden_states=True):
        # pixel_values: (B, T, C, H, W)
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=output_hidden_states)
        logits_fine = out.logits                      # (B, num_fine_classes)
        # cls token features from last hidden state
        cls_features = out.hidden_states[-1][:, 0, :]  # (B, hidden_size)
        logits_bin = self.binary_head(cls_features)    # (B, 2)
        return logits_fine, logits_bin

# ---------------------------
# Utilities: prepare model (DataParallel) & checkpoint helpers
# ---------------------------
def prepare_model(model, num_gpus=NUM_GPUS):
    if num_gpus <= 1:
        return model
    else:
        print(f"Using DataParallel on {num_gpus} GPUs")
        return nn.DataParallel(model)

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss
    }
    torch.save(payload, path)
    print(f"Saved checkpoint -> {path}")

def load_checkpoint_for_resume(path, model, optimizer=None, scheduler=None, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    start_epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"Resumed from checkpoint '{path}' (epoch {start_epoch}, best_val_loss={best_val_loss:.4f})")
    return start_epoch, best_val_loss

# ---------------------------
# Train / Eval functions
# ---------------------------
def train_one_epoch(model, loader, optimizer, scaler, crit_fine, crit_bin):
    model.train()
    running_loss = 0.0
    it = 0
    for videos, fine_labels, bin_labels in tqdm(loader, desc="Train", leave=False):
        # videos: (B, T, C, H, W)
        videos = videos.to(DEVICE, dtype=torch.float32)
        fine_labels = fine_labels.to(DEVICE)
        bin_labels = bin_labels.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            logits_fine, logits_bin = model(videos)
            loss_f = crit_fine(logits_fine, fine_labels)
            loss_b = crit_bin(logits_bin, bin_labels)
            loss = loss_f + loss_b

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        it += 1

    return running_loss / max(1, it)

def evaluate_model(model, loader, crit_fine=None, crit_bin=None):
    model.eval()
    total = 0
    correct_f = 0
    correct_b = 0
    running_loss = 0.0
    with torch.no_grad():
        for videos, fine_labels, bin_labels in tqdm(loader, desc="Eval", leave=False):
            videos = videos.to(DEVICE, dtype=torch.float32)
            fine_labels = fine_labels.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)

            logits_fine, logits_bin = model(videos)
            preds_f = logits_fine.argmax(dim=1)
            preds_b = logits_bin.argmax(dim=1)

            correct_f += (preds_f == fine_labels).sum().item()
            correct_b += (preds_b == bin_labels).sum().item()
            total += fine_labels.size(0)

            if crit_fine is not None and crit_bin is not None:
                running_loss += (crit_fine(logits_fine, fine_labels).item() + crit_bin(logits_bin, bin_labels).item())

    acc_f = correct_f / total if total > 0 else 0.0
    acc_b = correct_b / total if total > 0 else 0.0
    avg_loss = running_loss / (len(loader) if len(loader) > 0 else 1) if crit_fine is not None else None
    return acc_f, acc_b, avg_loss

# ---------------------------
# Main training driver
# ---------------------------
def train_model(model,
                train_loader, val_loader, test_loader,
                criterion, optimizer, scheduler,
                epochs, patience,
                device, checkpoint_dir,
                start_epoch=0, resume_path=None):
    scaler = GradScaler()
    best_val_loss = float("inf")
    best_metrics = None
    patience_counter = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    # if resuming checkpoint passed, load it
    if resume_path is not None and os.path.exists(resume_path):
        sepoch, best_val_loss = load_checkpoint_for_resume(resume_path, model, optimizer, scheduler, device)
        start_epoch = max(start_epoch, sepoch)

    print(f"Training from epoch {start_epoch} → {epochs} (total epochs = {epochs}), early-stop patience={patience}")

    crit_fine = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)  # label smoothing
    crit_bin = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, crit_fine, crit_bin)
        print(f"Train loss: {train_loss:.4f}")

        val_acc_f, val_acc_b, val_loss = evaluate_model(model, val_loader, crit_fine, crit_bin)
        print(f"Val fine-acc: {val_acc_f*100:.2f}%, val binary-acc: {val_acc_b*100:.2f}%, val_loss: {val_loss:.4f}")

        # scheduler step uses val_loss for ReduceLROnPlateau
        if scheduler is not None:
            scheduler.step(val_loss)

        # checkpoint on improvement (lower val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, CHECKPOINT_PATH)
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch + 1, best_val_loss)
            best_metrics = {"epoch": epoch + 1, "val_loss": best_val_loss, "val_acc_fine": val_acc_f, "val_acc_bin": val_acc_b}
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    # after training return best metrics
    return best_val_loss, best_metrics

# ---------------------------
# CLI / glue
# ---------------------------
def save_splits(train_entries, inner_val_entries, val_entries):
    # Each entry already is [abs_path, bin_label, fine_idx]
    data = {"train": train_entries, "inner_val": inner_val_entries, "val": val_entries}
    with open(SPLIT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved splits to {SPLIT_FILE} (count: train={len(train_entries)} inner_val={len(inner_val_entries)} val={len(val_entries)})")

def load_splits():
    with open(SPLIT_FILE, "r") as f:
        data = json.load(f)
    return data["train"], data["inner_val"], data["val"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (requires splits_vaibhav.json)")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint to resume from (optional)")
    args = parser.parse_args()

    # Build or load splits
    if args.resume:
        if not os.path.exists(SPLIT_FILE):
            raise RuntimeError(f"Resuming requested but splits file missing: {SPLIT_FILE}")
        train_entries, inner_val_entries, val_entries = load_splits()
        print(f"Loaded splits from {SPLIT_FILE}")
    else:
        all_entries = build_samples(DATA_ROOT)
        np.random.seed(42)
        np.random.shuffle(all_entries)
        n = len(all_entries)
        main_split = int(0.8 * n)
        train_main = all_entries[:main_split]
        val_entries = all_entries[main_split:]
        inner_split = int(0.9 * len(train_main))
        train_entries = train_main[:inner_split]
        inner_val_entries = train_main[inner_split:]
        save_splits(train_entries, inner_val_entries, val_entries)

    # Datasets & loaders
    train_ds = VideoFolderDataset(train_entries)
    inner_val_ds = VideoFolderDataset(inner_val_entries)
    val_ds = VideoFolderDataset(val_entries)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "collate_fn": collate_videos,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "prefetch_factor": PREFETCH_FACTOR
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    # inner_val and val should not shuffle
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_videos,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
    inner_val_loader = DataLoader(inner_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_videos,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)

    # Determine number of fine classes
    num_fine = max([entry[2] for entry in train_entries]) + 1

    # Build model
    model = VideoMAEWithBinary(num_fine)
    model = prepare_model(model, num_gpus=NUM_GPUS)
    model.to(DEVICE)

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR, verbose=True)

    resume_ckpt = args.resume_checkpoint if args.resume_checkpoint else (CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None)
    start_epoch = 0
    resume_path = None
    if args.resume and resume_ckpt is not None:
        resume_path = resume_ckpt
        print(f"Resuming from checkpoint: {resume_path}")

    # Train (will resume if resume_path provided)
    best_val_loss, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=inner_val_loader,    # note: inner_val used for scheduler and monitoring inside train loop. We'll evaluate final val inside loop too.
        test_loader=val_loader,         # not used inside train_model directly but kept for possible future usage
        criterion=None,                 # handled inside
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        checkpoint_dir=".",
        start_epoch=start_epoch,
        resume_path=resume_path
    )

    print("Training finished. Best metrics:", best_metrics)

if __name__ == "__main__":
    main()
