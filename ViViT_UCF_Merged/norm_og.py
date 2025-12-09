import os
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.io import read_video
import numpy as np

# ===========================
# CONFIGURATION
# ===========================
SOURCE_ROOT = "../UCF_MERGED/"           # contains "normal" and "abnormal"
DEST_ROOT   = "../NORM_UCF_MERGED_FPS/"    # output folder

TARGET_SIZE = (224, 224)  # (width, height)
TARGET_FPS = 12  # Target frame rate for sampling (12 FPS for 10s videos = 120 frames)
MAX_FRAMES = 128  # Maximum frames per video

# Empirical mean/std for UCF-Crime + UCF101 (literature values)
NORMALIZE_MEAN = [
    0.3763090481649224,
    0.36528133079917907,
    0.3509803885002057
  ]
NORMALIZE_STD  = [
    0.2708064877155682,
    0.266584700847205,
    0.26854724531039814
  ]

# ===========================
# VIDEO NORMALIZATION (resolution and FPS-based frame sampling)
# ===========================
def normalize_video_properties(src_path, dest_path):
    """Resize to target resolution and sample frames at TARGET_FPS rate, up to MAX_FRAMES."""
    try:
        clip = VideoFileClip(src_path)
        duration = clip.duration
        
        if duration <= 0:
            print(f"[ERROR] {src_path}: Invalid duration")
            return
        
        # Calculate target number of frames based on duration and target FPS
        target_frames = int(duration * TARGET_FPS)
        target_frames = min(target_frames, MAX_FRAMES)  # Cap at MAX_FRAMES
        
        if target_frames <= 0:
            print(f"[ERROR] {src_path}: No frames to extract")
            return
        
        # Sample frames at TARGET_FPS intervals
        # Ensure timestamps don't exceed video duration (subtract small epsilon)
        sample_times = np.arange(0, duration, 1.0 / TARGET_FPS)[:target_frames]
        sample_times = np.clip(sample_times, 0, duration - 1e-6)  # Clamp to valid range
        
        # Extract frames at sampled times
        sampled_frames = []
        for t in sample_times:
            try:
                frame = clip.get_frame(t)
                sampled_frames.append(frame)
            except Exception as e:
                print(f"[WARNING] Failed to extract frame at {t}s from {src_path}: {e}")
                continue
        
        if len(sampled_frames) == 0:
            print(f"[WARNING] {src_path}: No frames extracted")
            return
        
        # If we got fewer frames than target, pad with the last frame
        if len(sampled_frames) < target_frames:
            last_frame = sampled_frames[-1]
            padding_needed = target_frames - len(sampled_frames)
            sampled_frames.extend([last_frame] * padding_needed)
            print(f"[INFO] {src_path}: Padded {padding_needed} frames")
        
        # Create a new clip from sampled frames
        new_clip = ImageSequenceClip(sampled_frames, fps=TARGET_FPS)  # Use TARGET_FPS for output
        
        # Resize to target resolution
        new_clip = new_clip.resize(TARGET_SIZE)
        
        # Ensure dest folder
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Export normalized video
        new_clip.write_videofile(dest_path, codec="libx264", audio=False, verbose=False, logger=None)
        clip.close()
        new_clip.close()
        
        print(f"[SUCCESS] {src_path}: {len(sampled_frames)} frames at {TARGET_FPS} FPS")
        
    except Exception as e:
        print(f"[ERROR] {src_path}: {e}")

# ===========================
# FRAME NORMALIZATION (for training)
# ===========================
video_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

def load_and_normalize_frames(video_path):
    """Return a normalized tensor ready for model input."""
    video, _, _ = read_video(video_path, pts_unit='sec')
    # (T, H, W, C) -> (T, C, H, W)
    video = video.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(NORMALIZE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(1, 3, 1, 1)
    return (video - mean) / std

# ===========================
# MAIN PIPELINE
# ===========================
for label in ["Normal", "Abnormal"]:
    label_path = os.path.join(SOURCE_ROOT, label)
    if not os.path.isdir(label_path):
        continue

    for action_class in os.listdir(label_path):
        action_path = os.path.join(label_path, action_class)
        if not os.path.isdir(action_path):
            continue

        print(f"\n🎬 Processing {label}/{action_class}")
        dest_action_path = os.path.join(DEST_ROOT, label, action_class)
        os.makedirs(dest_action_path, exist_ok=True)

        for file in tqdm(os.listdir(action_path), desc=f"{label}/{action_class}"):
            if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            src_path = os.path.join(action_path, file)
            dest_path = os.path.join(dest_action_path, file)
            normalize_video_properties(src_path, dest_path)
