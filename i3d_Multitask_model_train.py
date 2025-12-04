# i3d_finetune_vivit_consistent.py
import os
import json
import random
from pathlib import Path
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import video
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------------
# User configuration / hyperparams
# -------------------------------
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 4
PIN_MEMORY = True
BENCHMARK = True
PREFETCH_FACTOR = 2
torch.backends.cudnn.benchmark = BENCHMARK

BATCH_SIZE = 8           # as requested
MIXED_PRECISION = True   # use AMP
USE_GPU_PRELOADING = False  # if True, loads all preprocessed videos to GPU (requires huge VRAM)

NUM_FRAMES = 16          # IMPORTANT: use 32 frames to match ViViT preprocessing
IMAGE_SIZE = 224

LR = 1e-4
WEIGHT_DECAY = 1e-3      # as requested
ADDITIONAL_EPOCHS = 200
PATIENCE = 20            # early stopping patience
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# -------------------------------
# Preprocessing (ViViT compatible)
# -------------------------------
vivit_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def sample_uniform(frames, num_frames):
    if len(frames) == 0:
        return []
    idx = np.linspace(0, len(frames)-1, num_frames).astype(int)
    return [frames[i] for i in idx]

def preprocess_video_cv2(video_path, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE):
    """Returns tensor of shape [1, T, C, H, W]"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(1, num_frames, 3, image_size, image_size, dtype=torch.float32)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        except Exception:
            continue
    cap.release()

    if len(frames) == 0:
        return torch.zeros(1, num_frames, 3, image_size, image_size, dtype=torch.float32)

    frames = sample_uniform(frames, num_frames)
    frames = torch.stack([vivit_transform(f) for f in frames])  # T, C, H, W
    return frames.unsqueeze(0)  # 1, T, C, H, W


# -------------------------------
# Dataset
# -------------------------------
class VideoDataset(Dataset):
    def __init__(self, samples, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, preload_gpu=False, device='cuda'):
        self.samples = samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.preload_gpu = preload_gpu
        self.device = device
        self._preloaded = None

        if self.preload_gpu:
            print("Preloading dataset to GPU (this may require a lot of VRAM)...")
            self._preloaded = []
            for p, b, f in tqdm(self.samples, desc="Preloading"):
                vid = preprocess_video_cv2(p, self.num_frames, self.image_size).squeeze(0)
                self._preloaded.append((vid.to(self.device), torch.tensor(b), torch.tensor(f)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._preloaded is not None:
            vid, b, f = self._preloaded[idx]
            return vid, b, f
        path, bin_label, fine_label = self.samples[idx]
        vid = preprocess_video_cv2(path, self.num_frames, self.image_size)  # 1,T,C,H,W
        vid = vid.squeeze(0)  # T,C,H,W
        return vid, torch.tensor(bin_label, dtype=torch.long), torch.tensor(fine_label, dtype=torch.long)


# -------------------------------
# Build samples (same folder structure)
# -------------------------------
def build_samples(root_dir):
    root = Path(root_dir)
    samples = []
    fine_classes = []

    for top in ["Normal", "Abnormal"]:
        d = root / top
        if not d.exists():
            continue
        for cls in sorted(x for x in d.iterdir() if x.is_dir()):
            fine_classes.append(cls.name)
            for ext in ["*.mp4", "*.avi", "*.mov"]:
                for p in cls.glob(ext):
                    bin_label = 0 if top == "Normal" else 1
                    samples.append((str(p), bin_label, cls.name))
    fine_classes = sorted(set(fine_classes))
    fine_to_idx = {c: i for i, c in enumerate(fine_classes)}
    final_samples = [(p, b, fine_to_idx[f]) for p, b, f in samples]
    return final_samples, fine_to_idx


# -------------------------------
# Multi-task I3D model
# -------------------------------
class MultiTaskI3D(nn.Module):
    def __init__(self, num_fine):
        super().__init__()
        self.backbone = video.mc3_18(weights="DEFAULT")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.binary_head = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 2)
        )
        self.fine_head = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, num_fine)
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> backbone expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        feats = self.backbone(x)
        return self.binary_head(feats), self.fine_head(feats)


# -------------------------------
# Helpers: prepare model & checkpoint resume
# -------------------------------
def prepare_model(model, num_gpus=1):
    if num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
    return model

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint_for_resume(model, optimizer, scheduler, ckpt_path, device):
    """
    Loads ckpt that contains:
    {
      'epoch': int,
      'model_state': state_dict,
      'optimizer_state': ...,
      'scheduler_state': ...,
      'best_val': float
    }
    Returns start_epoch (int) and best_val (float)
    """
    data = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(data['model_state'])
    if 'optimizer_state' in data and optimizer is not None:
        optimizer.load_state_dict(data['optimizer_state'])
    if 'scheduler_state' in data and scheduler is not None:
        try:
            scheduler.load_state_dict(data['scheduler_state'])
        except Exception:
            pass
    start_epoch = int(data.get('epoch', 0))
    best_val = float(data.get('best_val', float('inf')))
    return start_epoch, best_val


# -------------------------------
# Training loop (with AMP and LR scheduler)
# -------------------------------
def train_model(model, trainloader, valloader, testloader,
                criterion, optimizer, scheduler,
                epochs, early_stop_patience, device,
                checkpoints_dir, data_subset=None, start_epoch=0):
    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION and device.startswith("cuda"))
    best_val_loss = float('inf')
    best_epoch = -1
    no_improve = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_loss = 0.0
        total = 0
        correct_bin = 0
        correct_fine = 0
        t0 = time.time()

        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{start_epoch+epochs} [train]")
        for vid, yb, yf in pbar:
            # vid: (B, T, C, H, W)
            vid = vid.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yf = yf.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION and device.startswith("cuda")):
                out_b, out_f = model(vid)
                loss = criterion(out_b, yb) + criterion(out_f, yf)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = vid.size(0)
            train_loss += loss.item() * batch_size
            total += batch_size
            correct_bin += (out_b.argmax(1) == yb).sum().item()
            correct_fine += (out_f.argmax(1) == yf).sum().item()

            pbar.set_postfix({'loss': train_loss/total, 'bin_acc': 100.*correct_bin/total, 'fine_acc': 100.*correct_fine/total})

        t_elapsed = time.time() - t0
        train_loss /= max(total,1)
        train_bin_acc = 100.*correct_bin/total if total>0 else 0.0
        train_fine_acc = 100.*correct_fine/total if total>0 else 0.0

        # Validation
        val_loss, val_bin_acc, val_fine_acc = evaluate(model, valloader, criterion, device)
        # Scheduler expects a scalar to minimize (use val_loss)
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except Exception:
                # fallback if scheduler is ReduceLROnPlateau which expects .step(metric)
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1} summary: train_loss={train_loss:.4f}, train_bin={train_bin_acc:.2f}%, train_fine={train_fine_acc:.2f}%")
        print(f"                 val_loss={val_loss:.4f}, val_bin={val_bin_acc:.2f}%, val_fine={val_fine_acc:.2f}%  time={t_elapsed/60:.2f}m")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': getattr(scheduler, 'state_dict', lambda: None)(),
            'best_val': best_val_loss
        }
        ckpt_path = os.path.join(checkpoints_dir, f"ckpt_epoch_{epoch+1}.pth")
        save_checkpoint(ckpt, ckpt_path)

        # Best model by val_loss (min)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(checkpoints_dir, "best_i3d_multitask.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': getattr(scheduler, 'state_dict', lambda: None)(),
                'best_val': best_val_loss
            }, best_path)
            print(f"Saved BEST model at epoch {epoch+1} with val_loss={val_loss:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s)")

        # Early stopping
        if no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {no_improve} epochs without improvement.")
            break

    # test evaluation using best checkpoint if exists
    best_ckpt_path = os.path.join(checkpoints_dir, "best_i3d_multitask.pth")
    if os.path.exists(best_ckpt_path):
        print("Loading best checkpoint for final test evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    test_loss, test_bin_acc, test_fine_acc = evaluate(model, testloader, criterion, device)
    print("\nFINAL TEST RESULTS:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Binary Accuracy: {test_bin_acc:.2f}%")
    print(f"Fine Accuracy: {test_fine_acc:.2f}%")
    return best_val_loss, {'test_loss': test_loss, 'test_bin': test_bin_acc, 'test_fine': test_fine_acc}


# -------------------------------
# Evaluation helper
# -------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    run_loss = 0.0
    correct_bin = 0
    correct_fine = 0
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluating")
        for vid, yb, yf in pbar:
            vid = vid.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yf = yf.to(device, non_blocking=True)
            out_b, out_f = model(vid)
            loss = criterion(out_b, yb) + criterion(out_f, yf)
            batch = vid.size(0)
            run_loss += loss.item() * batch
            total += batch
            correct_bin += (out_b.argmax(1) == yb).sum().item()
            correct_fine += (out_f.argmax(1) == yf).sum().item()
    if total == 0:
        return float('inf'), 0.0, 0.0
    return run_loss / total, 100.*correct_bin/total, 100.*correct_fine/total


# -------------------------------
# Main entry point
# -------------------------------
def main():
    root = "/home/shubhranil/human_action_recognition/UCF_MERGED_NORM"
    split_file = "splits.json"

    # build dataset samples and class map
    samples, fine_map = build_samples(root)
    print("Videos:", len(samples))
    print("Fine classes:", len(fine_map))

    # load or create split
    if os.path.exists(split_file):
        print("Loading split from splits.json")
        data = json.load(open(split_file))
        train_s = data['train']
        val_s = data['val']
        test_s = data['test']
    else:
        print("Creating and saving new split (80/20 then 90/10 inside train pool)")
        random.shuffle(samples)
        N = len(samples)
        train_pool = samples[:int(0.8*N)]
        test_s = samples[int(0.8*N):]
        Tp = len(train_pool)
        train_s = train_pool[:int(0.9*Tp)]
        val_s = train_pool[int(0.9*Tp):]
        json.dump({'train': train_s, 'val': val_s, 'test': test_s}, open(split_file, 'w'))
        print("Saved splits.json")

    print("Sizes => train:", len(train_s), "val:", len(val_s), "test:", len(test_s))

    # datasets & dataloaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preload = USE_GPU_PRELOADING and device.startswith("cuda")
    train_ds = VideoDataset(train_s, preload_gpu=preload, device=device)
    val_ds = VideoDataset(val_s, preload_gpu=preload, device=device)
    test_ds = VideoDataset(test_s, preload_gpu=preload, device=device)

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
    valloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR)

    # model, criterion, optimizer, scheduler
    model = MultiTaskI3D(num_fine=len(fine_map))
    model = prepare_model(model, num_gpus=NUM_GPUS)
    model = model.to(device)

    # label smoothing
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    except TypeError:
        # older pytorch: fallback to normal crossentropy (no smoothing)
        print("Label smoothing not supported in this torch version; using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Use ReduceLROnPlateau (minimize val loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # resume checkpoint support (optional)
    resume_checkpoint = None  # <-- if you want to resume, set path here or pass via env/arg
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print("Resuming from checkpoint:", resume_checkpoint)
        start_epoch, best_val_loss = load_checkpoint_for_resume(model, optimizer, scheduler, resume_checkpoint, device)
        print(f"Resumed. Start epoch = {start_epoch}, best_val_loss = {best_val_loss}")

    # run training
    best_val, metrics = train_model(
        model, trainloader, valloader, testloader,
        criterion, optimizer, scheduler,
        epochs=ADDITIONAL_EPOCHS, early_stop_patience=PATIENCE,
        device=device, checkpoints_dir=CHECKPOINTS_DIR, data_subset=None, start_epoch=start_epoch
    )

if __name__ == "__main__":
    main()

