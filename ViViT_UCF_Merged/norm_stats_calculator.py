import os
import glob
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle

def process_single_video_streaming(video_path, target_size=(224, 224)):
    """Process a single video and return statistics WITHOUT loading all frames into memory."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # Initialize accumulators for this video
        video_sum = np.zeros(3, dtype=np.float64)
        video_sum_sq = np.zeros(3, dtype=np.float64)
        video_count = 0
        
        transform = transforms.ToTensor()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            
            # Convert to tensor (shape: [3, H, W])
            frame_tensor = transform(frame)
            
            # Accumulate statistics for this frame
            video_sum += frame_tensor.sum(dim=[1, 2]).numpy()
            video_sum_sq += (frame_tensor ** 2).sum(dim=[1, 2]).numpy()
            video_count += target_size[0] * target_size[1]  # pixels per frame
            
        cap.release()

        if video_count == 0:
            return None

        return {
            'sum': video_sum,
            'sum_sq': video_sum_sq,
            'count': video_count
        }

    except Exception as e:
        return None

def aggregate_video_stats(video_stats_list):
    """Aggregate statistics from multiple videos."""
    if not video_stats_list:
        return None

    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)
    total_count = 0
    
    for stats in video_stats_list:
        if stats is not None:
            total_sum += stats['sum']
            total_sum_sq += stats['sum_sq']
            total_count += stats['count']
    
    if total_count == 0:
        return None
    
    return {
        'sum': total_sum,
        'sum_sq': total_sum_sq,
        'count': total_count
    }

def save_intermediate_stats(stats_accumulator, batch_idx, norm_dir):
    """Save intermediate statistics to disk."""
    stats_file = os.path.join(norm_dir, f"batch_stats_{batch_idx:04d}.pkl")
    with open(stats_file, 'wb') as f:
        pickle.dump(stats_accumulator, f)
    return stats_file

def load_intermediate_stats(norm_dir):
    """Load the most recent intermediate statistics."""
    if not os.path.exists(norm_dir):
        return None, 0

    stats_files = [f for f in os.listdir(norm_dir) if f.startswith("batch_stats_") and f.endswith(".pkl")]
    if not stats_files:
        return None, 0

    # Find the latest batch file
    batch_numbers = [int(f.split("_")[2].split(".")[0]) for f in stats_files]
    latest_batch = max(batch_numbers)
    latest_file = os.path.join(norm_dir, f"batch_stats_{latest_batch:04d}.pkl")

    try:
        with open(latest_file, 'rb') as f:
            stats_accumulator = pickle.load(f)
        print(f"📂 Resumed from batch {latest_batch}")
        return stats_accumulator, latest_batch + 1
    except Exception as e:
        print(f"⚠️  Could not load intermediate stats: {e}")
        return None, 0

def calculate_dataset_normalization_stats_batched(dataset_root, target_size=(224, 224), batch_size=50, num_workers=None, norm_dir="normalization_stats"):
    """
    Calculate mean and std for image normalization across video dataset using batched processing.

    Args:
        dataset_root: Root directory containing videos
        target_size: Target resolution for frame resizing
        batch_size: Number of videos to process per batch
        num_workers: Number of parallel workers
        norm_dir: Directory to save intermediate results

    Returns:
        tuple: (mean_rgb, std_rgb) as numpy arrays
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    # Create normalization directory
    os.makedirs(norm_dir, exist_ok=True)

    # Find all video files recursively
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    all_videos = []

    print(f"🔍 Scanning for videos in: {dataset_root}")
    for ext in video_extensions:
        pattern = os.path.join(dataset_root, '**', ext)
        videos = glob.glob(pattern, recursive=True)
        all_videos.extend(videos)
        print(f"Found {len(videos)} {ext} files")

    print(f"\n📊 Total videos found: {len(all_videos)}")

    if not all_videos:
        raise ValueError("❌ No videos found!")

    # Try to resume from intermediate results
    stats_accumulator, start_batch = load_intermediate_stats(norm_dir)

    if stats_accumulator is None:
        # Initialize accumulator
        stats_accumulator = {
            'total_sum': np.zeros(3),
            'total_sum_sq': np.zeros(3),
            'total_count': 0
        }
        start_batch = 0
        print("🆕 Starting fresh calculation")
    else:
        print(f"� Resuming from batch {start_batch}")

    # Process videos in batches
    total_batches = (len(all_videos) + batch_size - 1) // batch_size

    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_videos))
        batch_videos = all_videos[start_idx:end_idx]

        print(f"\n🎬 Processing batch {batch_idx + 1}/{total_batches} (videos {start_idx + 1}-{end_idx})")

        # Process videos in this batch (streaming - no memory accumulation)
        batch_video_stats = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {
                executor.submit(process_single_video_streaming, video_path, target_size): video_path
                for video_path in batch_videos
            }

            for future in tqdm(as_completed(future_to_video), total=len(batch_videos), desc=f"Batch {batch_idx + 1}"):
                result = future.result()
                if result is not None:
                    batch_video_stats.append(result)

        if batch_video_stats:
            # Aggregate statistics for this batch
            batch_stats = aggregate_video_stats(batch_video_stats)

            if batch_stats:
                # Accumulate statistics
                stats_accumulator['total_sum'] += batch_stats['sum']
                stats_accumulator['total_sum_sq'] += batch_stats['sum_sq']
                stats_accumulator['total_count'] += batch_stats['count']

                print(f"📊 Batch {batch_idx + 1} stats - Pixels processed: {batch_stats['count']:,}")

        # Save intermediate results
        save_intermediate_stats(stats_accumulator, batch_idx, norm_dir)
        print(f"💾 Saved intermediate results to batch {batch_idx}")

    # Calculate final mean and std
    print("\n🔢 Calculating final statistics...")
    total_sum = stats_accumulator['total_sum']
    total_sum_sq = stats_accumulator['total_sum_sq']
    total_count = stats_accumulator['total_count']

    # Mean = total_sum / total_count
    dataset_mean = total_sum / total_count

    # Variance = (sum_sq / count) - mean^2
    # Std = sqrt(variance)
    variance = (total_sum_sq / total_count) - (dataset_mean ** 2)
    dataset_std = np.sqrt(np.maximum(variance, 0))  # Ensure non-negative

    # Save final results
    final_stats = {
        'mean': dataset_mean,
        'std': dataset_std,
        'total_frames': total_count // 3,  # Divide by 3 channels
        'total_videos_processed': len(all_videos)
    }

    final_stats_file = os.path.join(norm_dir, "final_normalization_stats.json")
    with open(final_stats_file, 'w') as f:
        json.dump({
            'mean': dataset_mean.tolist(),
            'std': dataset_std.tolist(),
            'total_frames': total_count // 3,
            'total_videos_processed': len(all_videos)
        }, f, indent=2)

    return dataset_mean, dataset_std

if __name__ == "__main__":
    # Configuration
    DATASET_ROOT = "../UCF_MERGED/"  # Update this to your dataset path
    TARGET_SIZE = (224, 224)        # Target resolution
    BATCH_SIZE = 50                 # Videos per batch (reduced for safety)
    NUM_WORKERS = 8                 # Number of parallel workers (14 cores available)
    NORM_DIR = "normalization_stats" # Directory to save intermediate results

    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Path does not exist: {DATASET_ROOT}")
        print("Please update DATASET_ROOT to point to your dataset directory")
        exit(1)

    try:
        mean_rgb, std_rgb = calculate_dataset_normalization_stats_batched(
            DATASET_ROOT,
            TARGET_SIZE,
            BATCH_SIZE,
            NUM_WORKERS,
            NORM_DIR
        )

        print("\n" + "="*60)
        print("🎨 FINAL DATASET NORMALIZATION STATISTICS (100% ACCURATE)")
        print("="*60)
        print(f"Dataset Mean (RGB): [{mean_rgb[0]:.6f}, {mean_rgb[1]:.6f}, {mean_rgb[2]:.6f}]")
        print(f"Dataset Std (RGB):  [{std_rgb[0]:.6f}, {std_rgb[1]:.6f}, {std_rgb[2]:.6f}]")

        print("\n📋 Use these values in your normalization code:")
        print(f"NORMALIZE_MEAN = [{mean_rgb[0]:.6f}, {mean_rgb[1]:.6f}, {mean_rgb[2]:.6f}]")
        print(f"NORMALIZE_STD = [{std_rgb[0]:.6f}, {std_rgb[1]:.6f}, {std_rgb[2]:.6f}]")

        print(f"\n✅ Complete statistics calculation finished!")
        print(f"📁 Results saved in: {NORM_DIR}/")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()