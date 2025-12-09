import os
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.io import read_video
import numpy as np
import json

# ===========================
# CONFIGURATION
# ===========================
SOURCE_ROOT = "../UCF_MERGED/Normal/Normal_Videos_for_Event_Recognition/"           # contains action class subfolders with videos
DEST_ROOT   = "../UCF_MERGED_NORM/Normal/Normal_Videos_for_Event_Recognition/"    # output folder

TARGET_SIZE = (224, 224)  # (width, height)
TARGET_NUM_FRAMES = 32  # Fixed number of frames per video

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

# Load missing videos from JSON
MISSING_VIDEOS_JSON = "./found_misses.json"
try:
    with open(MISSING_VIDEOS_JSON, 'r') as f:
        missing_data = json.load(f)
    missing_videos = set(missing_data.get("missing_videos", []))
    print(f"Loaded {len(missing_videos)} missing videos to process.")
except FileNotFoundError:
    print(f"Warning: {MISSING_VIDEOS_JSON} not found. Processing all videos.")
    missing_videos = None

# ===========================
# VIDEO NORMALIZATION (resolution and uniform frame sampling)
# ===========================
def normalize_video_properties(src_path, dest_path):
    """Resize to target resolution and sample exactly TARGET_NUM_FRAMES frames uniformly."""
    try:
        clip = VideoFileClip(src_path)
        total_frames = int(clip.fps * clip.duration)
        
        if total_frames <= 0:
            print(f"[ERROR] {src_path}: No frames detected")
            clip.close()
            return
        
        # Sample TARGET_NUM_FRAMES uniformly across the video
        # Avoid the very last frame which might be corrupted
        safe_total_frames = max(total_frames - 2, 1)
        sample_indices = np.linspace(0, safe_total_frames - 1, TARGET_NUM_FRAMES, dtype=int)
        sample_times = sample_indices / clip.fps
        
        # Extract frames at sampled times
        sampled_frames = []
        for t in sample_times:
            try:
                frame = clip.get_frame(t)
                sampled_frames.append(frame)
            except Exception as e:
                # Skip problematic frames silently and continue
                continue
        
        # If we couldn't extract enough frames, pad with the last successfully extracted frame
        if len(sampled_frames) < TARGET_NUM_FRAMES:
            if len(sampled_frames) == 0:
                print(f"[ERROR] {src_path}: Could not extract any frames")
                clip.close()
                return
            # Pad with duplicates of the last frame
            while len(sampled_frames) < TARGET_NUM_FRAMES:
                sampled_frames.append(sampled_frames[-1].copy())
            print(f"[INFO] {src_path}: Padded to {TARGET_NUM_FRAMES} frames (some frames duplicated)")
        
        # Create a new clip from sampled frames
        new_clip = ImageSequenceClip(sampled_frames, fps=30)  # Use 30 fps for output
        
        # Resize to target resolution
        new_clip = new_clip.resize(TARGET_SIZE)
        
        # Ensure dest folder
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Export normalized video
        new_clip.write_videofile(dest_path, codec="libx264", audio=False, verbose=False, logger=None)
        clip.close()
        new_clip.close()
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
print(f"\n🎬 Processing videos from {SOURCE_ROOT}")
dest_root = DEST_ROOT
os.makedirs(dest_root, exist_ok=True)

# Get list of videos to process
all_files = os.listdir(SOURCE_ROOT)
video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
if missing_videos is not None:
    video_files = [f for f in video_files if f in missing_videos]

print(f"Found {len(video_files)} videos to process.")

for file in tqdm(video_files, desc="Processing videos"):
    src_path = os.path.join(SOURCE_ROOT, file)
    dest_path = os.path.join(dest_root, file)
    normalize_video_properties(src_path, dest_path)
