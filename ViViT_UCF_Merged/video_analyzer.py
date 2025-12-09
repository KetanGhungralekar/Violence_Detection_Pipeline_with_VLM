import os
import csv
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import glob
import numpy as np
import cv2
import torch
from torchvision import transforms

def get_video_info(video_path):
    """Extract basic information from a video file."""
    try:
        clip = VideoFileClip(video_path)
        info = {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'duration': round(clip.duration, 2) if clip.duration else 0,
            'fps': round(clip.fps, 2) if clip.fps else 0,
            'width': clip.w if clip.w else 0,
            'height': clip.h if clip.h else 0,
            'resolution': f"{clip.w}x{clip.h}" if clip.w and clip.h else "Unknown",
            'status': 'OK'
        }
        clip.close()
        return info
    except Exception as e:
        return {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'duration': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'resolution': "Error",
            'status': f"Error: {str(e)}"
        }

def calculate_normalization_stats(video_path, num_samples=10):
    """Calculate mean and std for a single video by sampling frames."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            cap.release()
            return None
            
        # Sample frames evenly across the video
        sample_indices = np.linspace(0, max(frame_count - 1, 1), num_samples, dtype=int)
        
        for idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                continue
            if idx in sample_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # Resize to target size
                frames.append(frame)
                
        cap.release()
        
        if not frames:
            return None
            
        # Convert to tensor and calculate stats
        transform = transforms.ToTensor()
        frame_tensors = [transform(frame) for frame in frames]
        all_frames = torch.stack(frame_tensors)  # Shape: [num_samples, 3, 224, 224]
        
        # Calculate mean and std across all frames and spatial dimensions
        mean = all_frames.mean(dim=[0, 2, 3])  # Mean per channel
        std = all_frames.std(dim=[0, 2, 3])    # Std per channel
        
        return mean.numpy(), std.numpy()
        
    except Exception as e:
        return None

def analyze_dataset(root_path, output_csv="video_analysis.csv", calc_norm_stats=False):
    """Analyze all videos in the dataset and save to CSV."""
    
    # Find all video files recursively
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    all_videos = []

    print(f"🔍 Scanning for videos in: {root_path}")
    for ext in video_extensions:
        pattern = os.path.join(root_path, '**', ext)
        videos = glob.glob(pattern, recursive=True)
        all_videos.extend(videos)
        print(f"Found {len(videos)} {ext} files")

    print(f"\n📊 Total videos found: {len(all_videos)}")

    if not all_videos:
        print("❌ No videos found!")
        return

    # Analyze each video
    results = []
    print("\n🎬 Analyzing videos...")
    
    # For normalization stats calculation
    all_means = []
    all_stds = []

    for video_path in tqdm(all_videos, desc="Processing videos"):
        info = get_video_info(video_path)
        results.append(info)
        
        # Calculate normalization stats if requested
        if calc_norm_stats and info['status'] == 'OK':
            stats = calculate_normalization_stats(video_path)
            if stats:
                mean, std = stats
                all_means.append(mean)
                all_stds.append(std)

    # Save to CSV
    print(f"\n💾 Saving results to: {output_csv}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['path', 'filename', 'duration', 'fps', 'width', 'height', 'resolution', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(f"Total videos processed: {len(results)}")

    valid_videos = [r for r in results if r['status'] == 'OK']
    print(f"Successfully processed: {len(valid_videos)}")

    if valid_videos:
        durations = [r['duration'] for r in valid_videos]
        fps_values = [r['fps'] for r in valid_videos]

        print(f"Duration range: {min(durations):.1f} - {max(durations):.1f} seconds")
        print(f"Average duration: {sum(durations)/len(durations):.1f} seconds")
        print(f"FPS range: {min(fps_values):.1f} - {max(fps_values):.1f}")
        print(f"Average FPS: {sum(fps_values)/len(fps_values):.1f}")

        # Common resolutions
        resolutions = {}
        for r in valid_videos:
            res = r['resolution']
            resolutions[res] = resolutions.get(res, 0) + 1

        print("\n📐 Top 5 resolutions:")
        sorted_res = sorted(resolutions.items(), key=lambda x: x[1], reverse=True)
        for res, count in sorted_res[:5]:
            print(f"  {res}: {count} videos")

    # Print normalization statistics if calculated
    if calc_norm_stats and all_means and all_stds:
        print("\n🎨 Normalization Statistics (calculated from sampled frames):")
        dataset_mean = np.mean(all_means, axis=0)
        dataset_std = np.mean(all_stds, axis=0)
        
        print(f"Dataset Mean (RGB): [{dataset_mean[0]:.6f}, {dataset_mean[1]:.6f}, {dataset_mean[2]:.6f}]")
        print(f"Dataset Std (RGB):  [{dataset_std[0]:.6f}, {dataset_std[1]:.6f}, {dataset_std[2]:.6f}]")
        
        print("\n📋 Use these values in your normalization code:")
        print(f"NORMALIZE_MEAN = [{dataset_mean[0]:.6f}, {dataset_mean[1]:.6f}, {dataset_mean[2]:.6f}]")
        print(f"NORMALIZE_STD = [{dataset_std[0]:.6f}, {dataset_std[1]:.6f}, {dataset_std[2]:.6f}]")

    print(f"\n✅ Analysis complete! Results saved to {output_csv}")

if __name__ == "__main__":
    # Configuration
    DATASET_ROOT = "../UCF_MERGED/"  # Update this path as needed
    OUTPUT_CSV = "video_analysis.csv"
    CALC_NORM_STATS = False  # Set to True to calculate normalization statistics

    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Path does not exist: {DATASET_ROOT}")
        exit(1)

    analyze_dataset(DATASET_ROOT, OUTPUT_CSV, CALC_NORM_STATS)