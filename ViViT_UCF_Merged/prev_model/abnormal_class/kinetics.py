"""
Modified UCF Video Classification Training Script

Key Changes:
1. load_data() function now splits data from a single folder into 56% train, 14% val, 30% test
2. Added load_pretrained_weights() function to load weights from 101-class model to 14-class model
3. Model initialization includes loading pre-trained weights with proper layer filtering

Usage:
- Update the data_root path in load_data() to point to your dataset folder
- Update pretrained_weights_path in main() to point to your pre-trained model file
- The script will automatically handle the class mismatch between pre-trained and current models
"""

import os
import time
import logging
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from vivit import load_video

# ----------------------- Logging Setup -----------------------
log_filename = f'./training_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("Training started")

# Import HuggingFace Transformers for pretrained ViViT
try:
    from transformers import VivitForVideoClassification, VivitConfig
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ HuggingFace Transformers available - will use pretrained ViViT")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("❌ HuggingFace Transformers not available - falling back to custom ViViT")
    from vivit import ViViT

# Try to import libraries for PyTorch pretrained models (fallback)
try:
    import torch.hub
    TORCH_HUB_AVAILABLE = True
except ImportError:
    TORCH_HUB_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ----------------------- Performance Settings -----------------------
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 4
PIN_MEMORY = True
BENCHMARK = True
PREFETCH_FACTOR = 2
torch.backends.cudnn.benchmark = BENCHMARK
BATCH_SIZE = 8  # Reduced from 16 due to increased frames (16->32) doubling memory usage
MIXED_PRECISION = True
USE_GPU_PRELOADING = False  # Set to True to preload all videos to GPU (requires significant VRAM)

# ----------------------- Custom Dataset with Caching -----------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=32, image_size=224, cache_size=100):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size
        self.cache = {}
        self.cache_size = cache_size
        
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            video = self.cache[idx]
        else:
            video = load_video(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
            # Ensure video tensor has correct shape for ViViT
            # Expected: (frames, channels, height, width) -> will be permuted to (channels, frames, height, width) in model
            if video.ndim == 5 and video.shape[0] == 1:
                video = video.squeeze(0)
            if len(self.cache) >= self.cache_size:
                remove_key = list(self.cache.keys())[0]
                del self.cache[remove_key]
            self.cache[idx] = video
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# ----------------------- GPU Preloading Dataset -----------------------
class GPUVideoDataset(Dataset):
    """
    Dataset that preloads all videos to GPU memory for maximum training speed.
    WARNING: This requires significant GPU memory. Use only if you have enough VRAM.
    """
    def __init__(self, video_paths, labels, num_frames=32, image_size=224, device='cuda', preload=True):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size
        self.device = device
        self.videos_gpu = []
        self.labels_gpu = []
        
        if preload:
            logger.info(f"🚀 Preloading {len(video_paths)} videos to GPU memory...")
            self._preload_to_gpu()
            logger.info(f"✅ Successfully preloaded {len(self.videos_gpu)} videos to GPU")
    
    def _preload_to_gpu(self):
        """Preload all videos to GPU memory"""
        for idx, video_path in enumerate(tqdm(self.video_paths, desc="Loading videos to GPU")):
            try:
                video = load_video(video_path, num_frames=self.num_frames, image_size=self.image_size)
                # Ensure video tensor has correct shape
                if video.ndim == 5 and video.shape[0] == 1:
                    video = video.squeeze(0)
                
                # Move to GPU and store
                video_gpu = video.to(self.device, non_blocking=True)
                label_gpu = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
                
                self.videos_gpu.append(video_gpu)
                self.labels_gpu.append(label_gpu)
            except Exception as e:
                logger.warning(f"Failed to load video {video_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.videos_gpu)
    
    def __getitem__(self, idx):
        return self.videos_gpu[idx], self.labels_gpu[idx]

# ----------------------- Data Preparation -----------------------
def load_data(data_root="../../UCF_MERGED_NORM/Abnormal/"):
    """
    Load data from a single folder containing class subdirectories and split into 56/14/30 train/val/test
    """
    # Class to exclude
    EXCLUDE_CLASS = ""
    
    # Get all class names from the data root and sort alphabetically
    all_classes = []
    if os.path.exists(data_root):
        all_classes = [d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))]
    
    # Filter out the excluded class and sort alphabetically for consistent ordering
    data_subset = sorted([cls for cls in all_classes if cls != EXCLUDE_CLASS])
    
    if EXCLUDE_CLASS in all_classes:
        logger.info(f"Excluding class: {EXCLUDE_CLASS}")
    logger.info(f"Found {len(data_subset)} classes: {data_subset}")
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(data_subset)}
    
    # Load all videos from all classes
    all_videos = []
    all_labels = []
    
    for class_name in data_subset:
        class_dir = os.path.join(data_root, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Warning: Class directory {class_dir} does not exist!")
            continue
            
        video_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            if os.path.isfile(video_path):
                all_videos.append(video_path)
                all_labels.append(class_to_idx[class_name])
            else:
                logger.warning(f"Warning: Video file {video_path} does not exist!")
    
    logger.info(f"Loaded {len(all_videos)} total videos from {len(data_subset)} classes.")
    '''
    # Split data into train (90%), val (5%), test (5%)
    # First split: 95% train+val, 5% test
    train_val_videos, test_videos, train_val_labels, test_labels = train_test_split(
        all_videos, all_labels,
        test_size=0.20,
        random_state=42,
        stratify=all_labels
    )

    # Second split: from the 95%, split into 90% train and 5% val (which is 94.74%/5.26% of the 95%)
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        train_val_videos, train_val_labels,
        test_size=0.125, 
        random_state=42,
        stratify=train_val_labels
    )
    
    logger.info(f"Data split - Train: {len(train_videos)} ({len(train_videos)/len(all_videos)*100:.1f}%)")
    logger.info(f"Data split - Val: {len(val_videos)} ({len(val_videos)/len(all_videos)*100:.1f}%)")
    logger.info(f"Data split - Test: {len(test_videos)} ({len(test_videos)/len(all_videos)*100:.1f}%)")
    '''
    # Check if splits.json exists for reproducible splits
    splits_file = "splits.json"
    skip_split_creation = False
    if os.path.exists(splits_file):
        logger.info(f"📋 Loading existing data splits from {splits_file}")
        try:
            with open(splits_file, 'r') as f:
                splits_data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️  splits.json is corrupted or empty ({e}). Will create new splits.")
            os.remove(splits_file)  # Remove corrupted file
            splits_data = None
        else:
            # Validate that splits are compatible with current dataset
            if (splits_data.get('data_root') != data_root or
                splits_data.get('num_classes') != len(data_subset) or
                splits_data.get('class_names') != data_subset):
                logger.warning("⚠️  Existing splits don't match current dataset configuration!")
                logger.warning("   - Will create new splits")
                os.remove(splits_file)  # Remove incompatible splits
                splits_data = None
            else:
                train_videos = splits_data['train_videos']
                train_labels = splits_data['train_labels']
                val_videos = splits_data['val_videos']
                val_labels = splits_data['val_labels']
                test_videos = splits_data['test_videos']
                test_labels = splits_data['test_labels']

                logger.info("✅ Loaded reproducible data splits")
                skip_split_creation = True
    if not skip_split_creation:
        logger.info("🎲 Creating new data splits...")

        # Split data into train (72%), val (8%), test (20%)
        # First split: 80% train+val, 20% test
        train_val_videos, test_videos, train_val_labels, test_labels = train_test_split(
            all_videos, all_labels,
            test_size=0.20,
            random_state=42,
            stratify=all_labels
        )

        # Second split: from the 80%, split into 90% train and 10% val (which is 8% of total)
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            train_val_videos, train_val_labels,
            test_size=0.125,  # 0.125 * 0.80 = 0.10 (10% of the 80% = 8% of total)
            random_state=42,
            stratify=train_val_labels
        )

        # Save splits for reproducibility
        splits_data = {
            'train_videos': train_videos,
            'train_labels': train_labels,
            'val_videos': val_videos,
            'val_labels': val_labels,
            'test_videos': test_videos,
            'test_labels': test_labels,
            'created_at': datetime.now().isoformat(),
            'data_root': data_root,
            'num_classes': len(data_subset),
            'class_names': data_subset
        }

        with open(splits_file, 'w') as f:
            json.dump(splits_data, f, indent=2)
        logger.info(f"💾 Saved data splits to {splits_file} for reproducibility")

    logger.info(f"Data split - Train: {len(train_videos)} ({len(train_videos)/len(all_videos)*100:.1f}%)")
    logger.info(f"Data split - Val: {len(val_videos)} ({len(val_videos)/len(all_videos)*100:.1f}%)")
    logger.info(f"Data split - Test: {len(test_videos)} ({len(test_videos)/len(all_videos)*100:.1f}%)")
    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, data_subset

# ----------------------- Multi-GPU Support -----------------------
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        
    def forward(self, x):
        return self.module(x)

def prepare_model(model, num_gpus=1):
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)
    return model

# ----------------------- Utility Function for Saving Metrics -----------------------
def save_metrics(all_labels, all_preds, class_names, save_dir, mode='test', epoch=None):
    prefix = f"epoch_{epoch+1}_" if mode == 'val' and epoch is not None else ""
    metrics_dir = os.path.join(save_dir, f"{mode}_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Compute confusion matrix for per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)  # Handle classes with zero instances
    
    overall_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }
    
    per_class_metrics = {
        'accuracy': per_class_acc,
        'precision': precision_score(all_labels, all_preds, average=None, zero_division=0),
        'recall': recall_score(all_labels, all_preds, average=None, zero_division=0),
        'f1': f1_score(all_labels, all_preds, average=None, zero_division=0)
    }
    
    # Save to text file
    metrics_file = os.path.join(metrics_dir, f"{prefix}detailed_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("===== Overall Metrics =====\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
        f.write("\n===== Per-Class Metrics =====\n")
        f.write(f"{'Class':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write("-" * 74 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<30} {per_class_metrics['accuracy'][i]:>10.4f} "
                   f"{per_class_metrics['precision'][i]:>10.4f} {per_class_metrics['recall'][i]:>10.4f} "
                   f"{per_class_metrics['f1'][i]:>10.4f}\n")
    
    # Save to CSV file
    metrics_csv = os.path.join(metrics_dir, f"{prefix}metrics.csv")
    with open(metrics_csv, "w") as f:
        f.write("Metric,Value\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric},{value:.4f}\n")
        f.write("\nClass,Accuracy,Precision,Recall,F1-Score\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{per_class_metrics['accuracy'][i]:.4f},"
                   f"{per_class_metrics['precision'][i]:.4f},{per_class_metrics['recall'][i]:.4f},"
                   f"{per_class_metrics['f1'][i]:.4f}\n")
    
    # Generate per-class F1 score bar plot
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(class_names)), per_class_metrics['f1'])
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title(f'{mode.capitalize()} Per-Class F1 Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{prefix}per_class_f1.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate confusion matrix plot for validation and test
    if mode != 'train':
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{mode.capitalize()} Confusion Matrix')
        plt.savefig(os.path.join(metrics_dir, f"{prefix}confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return overall_metrics, per_class_metrics

# ----------------------- Optimized Training Function -----------------------
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs, class_names, save_dir):
    model.train()
    epoch_loss = 0.0
    total_count = 0
    all_train_preds = []
    all_train_labels = []
    batch_times = []
    
    start_time = time.time()
    for batch_idx, (videos_batch, labels_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")):
        batch_start = time.time()
        
        # Only transfer to device if not already on GPU (for non-GPU preloaded datasets)
        if not videos_batch.is_cuda:
            videos_batch = videos_batch.to(device, non_blocking=PIN_MEMORY)
        if not labels_batch.is_cuda:
            labels_batch = labels_batch.to(device, non_blocking=PIN_MEMORY)
        
        if MIXED_PRECISION:
            with autocast('cuda'):
                outputs = model(videos_batch)
                # Handle HuggingFace model output - extract logits
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels_batch)
            scaler.scale(loss).backward()
            # Add gradient clipping for regularization
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos_batch)
            # Handle HuggingFace model output - extract logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels_batch)
            loss.backward()
            # Add gradient clipping for regularization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)
        
        batch_size = labels_batch.size(0)
        epoch_loss += loss.item() * batch_size
        total_count += batch_size
        
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labels_np = labels_batch.detach().cpu().numpy()
        all_train_preds.extend(preds)
        all_train_labels.extend(labels_np)
        
        batch_times.append(time.time() - batch_start)
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - "
                        f"Loss: {loss.item():.4f}, Batch time: {batch_times[-1]:.4f}s")
    
    epoch_time = time.time() - start_time
    epoch_loss_avg = epoch_loss / total_count
    
    overall_metrics, per_class_metrics = save_metrics(all_train_labels, all_train_preds, class_names, save_dir, mode='train', epoch=epoch)
    
    avg_batch_time = sum(batch_times) / len(batch_times)
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s (avg batch: {avg_batch_time:.4f}s) - "
                f"Train Loss: {epoch_loss_avg:.4f}, "
                f"Train Acc: {overall_metrics['accuracy']:.4f}, "
                f"Train Prec (macro): {overall_metrics['precision_macro']:.4f}, "
                f"Train Rec (macro): {overall_metrics['recall_macro']:.4f}, "
                f"Train F1 (macro): {overall_metrics['f1_macro']:.4f}")
    
    return epoch_loss_avg, overall_metrics, per_class_metrics

# ----------------------- Validation Function -----------------------
def validate(model, val_loader, criterion, device, class_names, save_dir, epoch):
    model.eval()
    val_loss = 0.0
    val_count = 0
    val_preds = []
    val_labels = []
    inference_times = []  # Track inference time per video
    
    with torch.no_grad():
        for videos_batch, labels_batch in tqdm(val_loader, desc="Validation"):
            # Only transfer to device if not already on GPU (for non-GPU preloaded datasets)
            if not videos_batch.is_cuda:
                videos_batch = videos_batch.to(device, non_blocking=PIN_MEMORY)
            if not labels_batch.is_cuda:
                labels_batch = labels_batch.to(device, non_blocking=PIN_MEMORY)
            
            # Record inference time
            inference_start = time.time()
            if MIXED_PRECISION:
                with autocast('cuda'): # added cuda 10th april
                    outputs = model(videos_batch)
                    # Handle HuggingFace model output - extract logits
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = criterion(logits, labels_batch)
            else:
                outputs = model(videos_batch)
                # Handle HuggingFace model output - extract logits
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels_batch)
            inference_end = time.time()
            
            # Calculate per-video inference time
            batch_inference_time = (inference_end - inference_start) / videos_batch.size(0)
            inference_times.extend([batch_inference_time] * videos_batch.size(0))
                
            batch_size = labels_batch.size(0)
            val_loss += loss.item() * batch_size
            val_count += batch_size
            
            preds = logits.argmax(dim=1).cpu().numpy()
            val_labels.extend(labels_batch.cpu().numpy())
            val_preds.extend(preds)
            
    val_loss_avg = val_loss / val_count
    overall_metrics, per_class_metrics = save_metrics(val_labels, val_preds, class_names, save_dir, mode='val', epoch=epoch)
    
    # Save inference time statistics to CSV
    val_metrics_dir = os.path.join(save_dir, "val_metrics")
    os.makedirs(val_metrics_dir, exist_ok=True)
    
    inference_df = pd.DataFrame({
        'video_index': range(len(inference_times)),
        'inference_time_seconds': inference_times,
        'predicted_class': val_preds,
        'true_class': val_labels
    })
    
    inference_csv_path = os.path.join(val_metrics_dir, f"epoch_{epoch+1}_inference_times.csv")
    inference_df.to_csv(inference_csv_path, index=False)
    
    # Log inference time statistics
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)
    
    logger.info(f"Validation - Loss: {val_loss_avg:.4f}, "
                f"Acc: {overall_metrics['accuracy']:.4f}, "
                f"Prec (macro): {overall_metrics['precision_macro']:.4f}, "
                f"Rec (macro): {overall_metrics['recall_macro']:.4f}, "
                f"F1 (macro): {overall_metrics['f1_macro']:.4f}")
    logger.info(f"Inference Time - Avg: {avg_inference_time:.4f}s, "
                f"Min: {min_inference_time:.4f}s, "
                f"Max: {max_inference_time:.4f}s, "
                f"Std: {std_inference_time:.4f}s")
    
    return val_loss_avg, overall_metrics, per_class_metrics

# ----------------------- Test Evaluation Function -----------------------
def evaluate_test(model, test_loader, device, class_names, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []  # Track inference time per video
    
    with torch.no_grad():
        for videos_batch, labels_batch in tqdm(test_loader, desc="Testing"):
            # Only transfer to device if not already on GPU (for non-GPU preloaded datasets)
            if not videos_batch.is_cuda:
                videos_batch = videos_batch.to(device, non_blocking=PIN_MEMORY)
            if not labels_batch.is_cuda:
                labels_batch = labels_batch.to(device, non_blocking=PIN_MEMORY)
            
            # Record inference time
            inference_start = time.time()
            with autocast('cuda', enabled=MIXED_PRECISION):
                outputs = model(videos_batch)
            inference_end = time.time()
            
            # Calculate per-video inference time
            batch_inference_time = (inference_end - inference_start) / videos_batch.size(0)
            inference_times.extend([batch_inference_time] * videos_batch.size(0))
            
            # Handle HuggingFace model output - extract logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    
    overall_metrics, per_class_metrics = save_metrics(all_labels, all_preds, class_names, save_dir, mode='test')
    
    # Save inference time statistics to CSV
    test_metrics_dir = os.path.join(save_dir, "test_metrics")
    os.makedirs(test_metrics_dir, exist_ok=True)
    
    inference_df = pd.DataFrame({
        'video_index': range(len(inference_times)),
        'inference_time_seconds': inference_times,
        'predicted_class': all_preds,
        'true_class': all_labels
    })
    
    inference_csv_path = os.path.join(test_metrics_dir, "test_inference_times.csv")
    inference_df.to_csv(inference_csv_path, index=False)
    
    # Log inference time statistics
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)
    
    logger.info(f"Test Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision (macro): {overall_metrics['precision_macro']:.4f}")
    logger.info(f"Test Recall (macro): {overall_metrics['recall_macro']:.4f}")
    logger.info(f"Test F1 Score (macro): {overall_metrics['f1_macro']:.4f}")
    logger.info(f"Test Precision (micro): {overall_metrics['precision_micro']:.4f}")
    logger.info(f"Test Recall (micro): {overall_metrics['recall_micro']:.4f}")
    logger.info(f"Test F1 Score (micro): {overall_metrics['f1_micro']:.4f}")
    logger.info(f"Test Precision (weighted): {overall_metrics['precision_weighted']:.4f}")
    logger.info(f"Test Recall (weighted): {overall_metrics['recall_weighted']:.4f}")
    logger.info(f"Test F1 Score (weighted): {overall_metrics['f1_weighted']:.4f}")
    logger.info(f"Inference Time - Avg: {avg_inference_time:.4f}s, "
                f"Min: {min_inference_time:.4f}s, "
                f"Max: {max_inference_time:.4f}s, "
                f"Std: {std_inference_time:.4f}s")
    
    return overall_metrics, per_class_metrics

# ----------------------- Comprehensive Test Evaluation Function -----------------------
def evaluate_test_comprehensive(model, test_loader, device, class_names, save_dir):
    """
    Comprehensive test evaluation with detailed metrics and visualizations
    """
    model.eval()
    all_preds = []
    all_labels = []
    inference_times = []  # Track inference time per video
    
    logger.info("🧪 Evaluating on test set...")
    with torch.no_grad():
        for videos_batch, labels_batch in tqdm(test_loader, desc="Testing"):
            # Only transfer to device if not already on GPU (for non-GPU preloaded datasets)
            if not videos_batch.is_cuda:
                videos_batch = videos_batch.to(device, non_blocking=PIN_MEMORY)
            if not labels_batch.is_cuda:
                labels_batch = labels_batch.to(device, non_blocking=PIN_MEMORY)
            
            # Record inference time
            inference_start = time.time()
            with autocast('cuda', enabled=MIXED_PRECISION):
                outputs = model(videos_batch)
            inference_end = time.time()
            
            # Calculate per-video inference time
            batch_inference_time = (inference_end - inference_start) / videos_batch.size(0)
            inference_times.extend([batch_inference_time] * videos_batch.size(0))
            
            # Handle HuggingFace model output - extract logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    
    # Overall metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Class-wise metrics
    classwise_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    classwise_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    classwise_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Print results
    logger.info(f"\n🎯 Test Results:")
    logger.info(f"  Overall Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Overall Precision: {test_precision:.4f}")
    logger.info(f"  Overall Recall: {test_recall:.4f}")
    logger.info(f"  Overall F1 Score: {test_f1:.4f}")
    
    # Create comprehensive test results directory
    test_dir = os.path.join(save_dir, "comprehensive_test_results")
    os.makedirs(test_dir, exist_ok=True)
    
    # Save inference time statistics to CSV
    inference_df = pd.DataFrame({
        'video_index': range(len(inference_times)),
        'inference_time_seconds': inference_times,
        'predicted_class': all_preds,
        'true_class': all_labels,
        'class_name_predicted': [class_names[pred] for pred in all_preds],
        'class_name_true': [class_names[label] for label in all_labels]
    })
    
    inference_csv_path = os.path.join(test_dir, "comprehensive_test_inference_times.csv")
    inference_df.to_csv(inference_csv_path, index=False)
    
    # Calculate and log inference time statistics
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)
    median_inference_time = np.median(inference_times)
    
    logger.info(f"\n⏱️  Inference Time Statistics:")
    logger.info(f"  Average: {avg_inference_time:.4f}s")
    logger.info(f"  Median: {median_inference_time:.4f}s")
    logger.info(f"  Min: {min_inference_time:.4f}s")
    logger.info(f"  Max: {max_inference_time:.4f}s")
    logger.info(f"  Std Dev: {std_inference_time:.4f}s")
    
    # Create inference time distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(avg_inference_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_inference_time:.4f}s')
    plt.axvline(median_inference_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_inference_time:.4f}s')
    plt.xlabel('Inference Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Per-Video Inference Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, "inference_time_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix on Test Set", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, "confusion_matrix_detailed.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class-wise performance bar chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(len(class_names))
    
    ax1.bar(x_pos, classwise_precision, color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Precision')
    ax1.set_title('Class-wise Precision')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x_pos, classwise_recall, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Recall')
    ax2.set_title('Class-wise Recall')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    ax3.bar(x_pos, classwise_f1, color='lightblue', alpha=0.7)
    ax3.set_xlabel('Class')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Class-wise F1 Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, "classwise_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed classification report
    try:
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(test_dir, "classification_report_detailed.csv"))
        logger.info("📊 Classification report saved as CSV")
    except Exception as e:
        logger.warning(f"Could not save classification report as CSV: {e}")
        # Fallback: save as text file
        with open(os.path.join(test_dir, "classification_report_detailed.txt"), "w") as f:
            f.write(classification_report(all_labels, all_preds, target_names=class_names))
    
    logger.info("📊 Comprehensive test evaluation plots and reports saved:")
    logger.info(f"  - {test_dir}/confusion_matrix_detailed.png")
    logger.info(f"  - {test_dir}/classwise_performance.png") 
    logger.info(f"  - {test_dir}/classification_report_detailed.csv")
    logger.info(f"  - {test_dir}/comprehensive_test_inference_times.csv")
    logger.info(f"  - {test_dir}/inference_time_distribution.png")
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'classwise_precision': classwise_precision,
        'classwise_recall': classwise_recall,
        'classwise_f1': classwise_f1,
        'inference_times': {
            'avg': avg_inference_time,
            'median': median_inference_time,
            'min': min_inference_time,
            'max': max_inference_time,
            'std': std_inference_time
        }
    }, None

# ----------------------- Main Training Loop -----------------------
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, 
                scheduler, num_epochs, patience, device, checkpoints_dir, class_names, start_epoch=0):
    
    scaler = GradScaler('cuda') if MIXED_PRECISION else None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    metrics_history = {
        "train_loss": [], "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [],
        "classwise_precision": [], "classwise_recall": [], "classwise_f1": []
    }
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_metrics, _ = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, 
            start_epoch + num_epochs, class_names, checkpoints_dir
        )
        
        val_loss, val_metrics, val_per_class_metrics = validate(
            model, val_loader, criterion, device, class_names, checkpoints_dir, epoch
        )
        
        # Update scheduler with validation loss (ReduceLROnPlateau)
        if scheduler:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.7f}")
        
        metrics_history["train_loss"].append(train_loss)
        metrics_history["train_accuracy"].append(train_metrics['accuracy'])
        metrics_history["train_precision"].append(train_metrics['precision_macro'])
        metrics_history["train_recall"].append(train_metrics['recall_macro'])
        metrics_history["train_f1"].append(train_metrics['f1_macro'])
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_metrics['accuracy'])
        metrics_history["val_precision"].append(val_metrics['precision_macro'])
        metrics_history["val_recall"].append(val_metrics['recall_macro'])
        metrics_history["val_f1"].append(val_metrics['f1_macro'])
        
        # Store class-wise metrics for plotting evolution
        metrics_history["classwise_precision"].append(val_per_class_metrics['precision'])
        metrics_history["classwise_recall"].append(val_per_class_metrics['recall'])
        metrics_history["classwise_f1"].append(val_per_class_metrics['f1'])
        
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
            
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_checkpoint_path)
            logger.info(f"New best model saved at: {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
    total_training_time = time.time() - total_start_time
    logger.info(f"Total training time: {total_training_time/60:.2f} minutes")
    
    model = load_best_model(model, os.path.join(checkpoints_dir, "best_checkpoint.pth"), device)
    test_metrics, _ = evaluate_test_comprehensive(model, test_loader, device, class_names, checkpoints_dir)
    
    return best_val_loss, metrics_history

# ----------------------- Utility Functions -----------------------
def load_pretrained_weights_pytorch(model, device='cuda'):
    """
    Load pre-trained ViViT weights from PyTorch Hub or timm for Kinetics dataset
    """
    logger.info("🔍 Attempting to load PyTorch pre-trained ViViT weights...")
    
    # Try torch.hub first
    if TORCH_HUB_AVAILABLE:
        try:
            logger.info("📥 Trying to load from PyTorch Hub...")
            # Try loading ViViT models from torch hub
            available_models = torch.hub.list('pytorch/vision', force_reload=False)
            vivit_models = [m for m in available_models if 'vivit' in m.lower()]
            
            if vivit_models:
                logger.info(f"Found ViViT models: {vivit_models}")
                for model_name in vivit_models:
                    try:
                        logger.info(f"Attempting to load {model_name}...")
                        pretrained_model = torch.hub.load('pytorch/vision', model_name, pretrained=True)
                        
                        # Transfer weights (excluding classifier layer)
                        model_dict = model.state_dict()
                        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                                         if k in model_dict and 'head' not in k and 'classifier' not in k}
                        
                        if pretrained_dict:
                            model_dict.update(pretrained_dict)
                            model.load_state_dict(model_dict, strict=False)
                            logger.info(f"✅ Successfully loaded PyTorch Hub weights from {model_name}")
                            return True
                    except Exception as e:
                        logger.info(f"❌ Failed to load {model_name}: {e}")
                        continue
        except Exception as e:
            logger.info(f"❌ PyTorch Hub loading failed: {e}")
    
    # Try timm as fallback
    if TIMM_AVAILABLE:
        try:
            logger.info("📥 Trying timm models...")
            vivit_models = timm.list_models('*vivit*')
            video_models = timm.list_models('*video*')
            
            all_models = vivit_models + video_models
            logger.info(f"Found timm models: {all_models}")
            
            for model_name in all_models:
                try:
                    logger.info(f"Attempting to load {model_name}...")
                    pretrained_model = timm.create_model(model_name, pretrained=True)
                    
                    # Transfer weights (excluding classifier layer)
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                                     if k in model_dict and 'head' not in k and 'classifier' not in k}
                    
                    if pretrained_dict:
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict, strict=False)
                        logger.info(f"✅ Successfully loaded timm weights from {model_name}")
                        return True
                except Exception as e:
                    logger.info(f"❌ Failed to load {model_name}: {e}")
                    continue
        except Exception as e:
            logger.info(f"❌ timm loading failed: {e}")
    
    logger.warning("❌ No PyTorch pre-trained weights found. Will train from scratch.")
    return False

def load_pretrained_weights(model, pretrained_path=None, num_classes_pretrained=101, num_classes_current=14):
    """
    Load pre-trained weights from a local file or PyTorch models
    """
    # First try to load PyTorch pretrained weights
    if load_pretrained_weights_pytorch(model):
        return model
    
    # Fallback to local file if provided
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pre-trained weights from local file: {pretrained_path}")
        
        # Load the pre-trained state dict
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        
        # Get current model state dict
        current_state = model.state_dict()
        
        # Filter out classifier layer weights if classes don't match
        filtered_state = {}
        for key, value in pretrained_state.items():
            if 'head' in key or 'classifier' in key or 'fc' in key:
                # Skip classifier weights if different number of classes
                if value.shape != current_state[key].shape:
                    logger.info(f"Skipping {key} due to shape mismatch: {value.shape} vs {current_state[key].shape}")
                    continue
            filtered_state[key] = value
        
        # Load the filtered weights
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        
        if missing_keys:
            logger.info(f"Missing keys (will use random initialization): {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys (ignored): {unexpected_keys}")
        
        logger.info("Pre-trained weights loaded successfully from local file!")
        return model
    elif pretrained_path:
        logger.warning(f"Pre-trained weights file not found: {pretrained_path}")
    
    logger.info("Training from scratch - no pre-trained weights loaded.")
    return model

def load_best_model(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {checkpoint_path}")
    return model

def find_latest_checkpoint(base_dir='.'):
    """
    Automatically find the most recent checkpoint directory and the latest checkpoint within it.
    Returns: (checkpoint_dir, checkpoint_path, start_epoch) or (None, None, 0) if no checkpoint found
    """
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('checkpoints_')]
    
    if not checkpoint_dirs:
        logger.info("No existing checkpoint directories found.")
        return None, None, 0
    
    # Sort by modification time to get the most recent
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    latest_dir = os.path.join(base_dir, checkpoint_dirs[0])
    
    logger.info(f"📂 Found checkpoint directory: {latest_dir}")
    
    # Look for the best checkpoint first, then epoch checkpoints
    best_checkpoint = os.path.join(latest_dir, "best_checkpoint.pth")
    
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"✅ Found best checkpoint at epoch {start_epoch}")
        return latest_dir, best_checkpoint, start_epoch
    
    # Find all epoch checkpoints
    epoch_checkpoints = [f for f in os.listdir(latest_dir) 
                        if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    
    if not epoch_checkpoints:
        logger.info("No checkpoint files found in directory.")
        return latest_dir, None, 0
    
    # Extract epoch numbers and find the latest
    def extract_epoch(filename):
        try:
            return int(filename.replace('checkpoint_epoch_', '').replace('.pth', ''))
        except:
            return 0
    
    epoch_checkpoints.sort(key=extract_epoch, reverse=True)
    latest_checkpoint = os.path.join(latest_dir, epoch_checkpoints[0])
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    start_epoch = checkpoint.get('epoch', 0)
    
    logger.info(f"✅ Found latest checkpoint: {epoch_checkpoints[0]} (epoch {start_epoch})")
    
    return latest_dir, latest_checkpoint, start_epoch

def load_checkpoint_for_resume(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load checkpoint and restore model, optimizer, and scheduler states for resuming training.
    Returns: start_epoch
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
        return 0
    
    logger.info(f"📥 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', 0.0)
    
    logger.info(f"✅ Successfully loaded checkpoint from epoch {start_epoch}")
    logger.info(f"   - Train Loss: {train_loss:.4f}")
    logger.info(f"   - Val Loss: {val_loss:.4f}")
    
    return start_epoch

def plot_metrics(metrics, save_dir='.'):
    epochs = range(1, len(metrics["train_loss"]) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_loss"], 'b-', label='Train Loss')
    plt.plot(epochs, metrics["val_loss"], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_accuracy"], 'b-', label='Train Accuracy')
    plt.plot(epochs, metrics["val_accuracy"], 'r-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_precision"], 'b-', label='Train Precision')
    plt.plot(epochs, metrics["val_precision"], 'r-', label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "precision_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_recall"], 'b-', label='Train Recall')
    plt.plot(epochs, metrics["val_recall"], 'r-', label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "recall_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_f1"], 'b-', label='Train F1 Score')
    plt.plot(epochs, metrics["val_f1"], 'r-', label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "f1_plot.png"), dpi=300)
    plt.close()

def plot_classwise_evolution(metrics, class_names, save_dir='.'):
    """
    Plot class-wise F1 score evolution over epochs
    """
    if not metrics["classwise_f1"] or len(metrics["classwise_f1"]) == 0:
        logger.warning("No class-wise metrics available for plotting evolution")
        return
    
    epochs = range(1, len(metrics["classwise_f1"]) + 1)
    
    # Set style for better looking plots
    plt.style.use('default')
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    plt.figure(figsize=(14, 8))
    
    # Convert class-wise metrics to arrays for easier plotting
    classwise_f1_array = np.array(metrics["classwise_f1"])  # Shape: (epochs, num_classes)
    
    # Plot evolution of each class's F1 score
    for class_idx, class_name in enumerate(class_names):
        if class_idx < classwise_f1_array.shape[1]:  # Safety check
            plt.plot(epochs, classwise_f1_array[:, class_idx] * 100, 
                    label=class_name, color=colors[class_idx], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (%)', fontsize=12)
    plt.title('Class-wise F1 Score Evolution', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, color='#E5E5E5')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "classwise_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("📊 Class-wise evolution plot saved: classwise_evolution.png")

# ----------------------- Main Execution -----------------------
def main():
    global TRANSFORMERS_AVAILABLE

    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")
    
    # Auto-detect and resume from existing checkpoint
    resume_dir, resume_checkpoint, resume_epoch = find_latest_checkpoint()
    
    if resume_checkpoint:
        # Use existing checkpoint directory
        checkpoints_dir = resume_dir
        logger.info(f"🔄 Resuming training from checkpoint directory: {checkpoints_dir}")
    else:
        # Create new checkpoint directory
        checkpoints_dir = f"./checkpoints_{datetime.now().strftime('%Y%m%d_%H%M')}"
        logger.info(f"🆕 Creating new checkpoints directory: {checkpoints_dir}")
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Load and split data
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, data_subset = load_data()
    
    logger.info("Creating datasets...")
    
    # Use GPU preloading if enabled and GPU is available
    if USE_GPU_PRELOADING and torch.cuda.is_available():
        logger.info("🚀 GPU preloading enabled - loading datasets to GPU memory...")
        train_dataset = GPUVideoDataset(train_videos, train_labels, device=device)
        val_dataset = GPUVideoDataset(val_videos, val_labels, device=device)
        test_dataset = GPUVideoDataset(test_videos, test_labels, device=device)
        # Disable pin_memory and reduce workers for GPU preloaded data
        use_pin_memory = False
        use_num_workers = 0
    else:
        logger.info("📦 Using standard dataset with CPU caching...")
        train_dataset = VideoDataset(train_videos, train_labels)
        val_dataset = VideoDataset(val_videos, val_labels)
        test_dataset = VideoDataset(test_videos, test_labels)
        use_pin_memory = PIN_MEMORY
        use_num_workers = NUM_WORKERS
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    trainloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=use_num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if use_num_workers > 0 else None,
        persistent_workers=True if use_num_workers > 0 else False
    )
    
    valloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=use_num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if use_num_workers > 0 else None,
        persistent_workers=True if use_num_workers > 0 else False
    )
    
    testloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=use_num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=PREFETCH_FACTOR if use_num_workers > 0 else None,
        persistent_workers=True if use_num_workers > 0 else False
    )
    
    logger.info("Initializing model...")
    
    if TRANSFORMERS_AVAILABLE:
        # Use HuggingFace pretrained ViViT
        logger.info("🚀 Using HuggingFace Transformers - Loading Google's pretrained ViViT from Kinetics-400")
        try:
            # Load the configuration
            config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
            
            # Print original configuration for verification
            logger.info(f"📋 Original ViViT config:")
            logger.info(f"   - num_frames: {config.num_frames}")
            logger.info(f"   - image_size: {config.image_size}")
            logger.info(f"   - tubelet_size: {config.tubelet_size}")
            logger.info(f"   - patch_size (spatial): {config.tubelet_size[1]}x{config.tubelet_size[2]}")
            logger.info(f"   - tubelet_size (temporal): {config.tubelet_size[0]}")
            
            # Modify the configuration for our number of classes
            num_classes = len(data_subset)
            config.num_labels = num_classes
            logger.info(f"🎯 Adapted classifier head for {num_classes} classes")
            
            # Load the pretrained model
            model = VivitForVideoClassification.from_pretrained(
                "google/vivit-b-16x2-kinetics400", 
                config=config,
                ignore_mismatched_sizes=True  # This allows loading with different num_classes
            )
            
            logger.info(f"✅ Successfully loaded pretrained ViViT-B-16x2 model")
            logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load HuggingFace model: {e}")
            logger.info("🔄 Falling back to custom ViViT implementation")
            TRANSFORMERS_AVAILABLE = False
    
    if not TRANSFORMERS_AVAILABLE:
        # Use custom ViViT implementation
        logger.info("📦 Using custom ViViT implementation")
        num_classes = len(data_subset)
        model = ViViT(image_size=224, patch_size=16, num_classes=num_classes, num_frames=32,
                      dropout=0.2, emb_dropout=0.1)
        # Try to load pre-trained weights for custom ViViT
        pretrained_weights_path = None  # Set to local path if you have one
        model = load_pretrained_weights(model, pretrained_weights_path, num_classes_pretrained=101, num_classes_current=num_classes)

        
        logger.info("Freezing initial model layers...")

        # 1. Freeze patch embedding
        for param in model.to_patch_embedding.parameters():
            param.requires_grad = False
        
        # 2. Freeze positional embedding and tokens (they are nn.Parameter)
        model.pos_embedding.requires_grad = False
        model.space_token.requires_grad = False
        model.temporal_token.requires_grad = False
        
        logger.info(" - Embeddings, pos_embedding, and tokens frozen.")

        # 3. Freeze the entire Space Transformer
        # This is equivalent to freezing the first 'depth' layers of the HF model
        for param in model.space_transformer.parameters():
            param.requires_grad = False
        
        # Note: The 'depth' in your ViViT class is 4 by default
        logger.info(f" - Entire Space Transformer frozen.")
        
        # 4. (Optional) Freeze the first N layers of the Temporal Transformer
        #    This is similar to your request to freeze 7 layers (4 spatial + 3 temporal)
        
        # num_temporal_layers_to_freeze = 3 
        # for i in range(num_temporal_layers_to_freeze):
        #     for param in model.temporal_transformer.layers[i].parameters():
        #         param.requires_grad = False
        # logger.info(f" - First {num_temporal_layers_to_freeze} Temporal Transformer layers frozen.")

        # 5. Verify which parameters are still trainable
        logger.info("\n--- Trainable Parameters ---")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
        logger.info("--------------------------\n")
        
        # --- END OF FREEZING CODE ---

    
    model = prepare_model(model, num_gpus=NUM_GPUS)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing for regularization
    # Increased weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)  # Starting with 1e-3 as requested
    # ReduceLROnPlateau scheduler with patience=10, factor=0.5 as requested
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Automatic checkpoint loading and resume
    start_epoch = 0
    
    if resume_checkpoint:
        logger.info("🔄 Resuming training from checkpoint...")
        start_epoch = load_checkpoint_for_resume(model, optimizer, scheduler, resume_checkpoint, device)
        logger.info(f"📍 Training will resume from epoch {start_epoch + 1}")
    else:
        logger.info("🆕 Starting training from scratch (no checkpoint found).")
    
    logger.info("Starting training...")
    additional_epochs = 200
    patience = 20
    best_val_loss, metrics = train_model(
        model, trainloader, valloader, testloader, 
        criterion, optimizer, scheduler, 
        additional_epochs, patience, device, 
        checkpoints_dir, data_subset, start_epoch
    )
    
    plot_metrics(metrics, save_dir=checkpoints_dir)
    plot_classwise_evolution(metrics, data_subset, save_dir=checkpoints_dir)
    logger.info(f"Training metrics plots saved to {checkpoints_dir}")
    
    final_model_path = os.path.join(checkpoints_dir, "vivit_model_final.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as {final_model_path}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
