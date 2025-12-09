import pandas as pd
from pathlib import Path
from collections import defaultdict
import time

# --- User Configuration ---
# PLEASE UPDATE THESE PATHS AND CLASSES BEFORE RUNNING

# List of classes to consider. Only videos from these classes will be processed.
# Ignore videos from other classes.
# 'Abuse', 'Arrest', 'Assault', 'Arson', 'Explosion', 
CLASSES_TO_CONSIDER = ['Fighting', 'RoadAccidents', 'Shooting', 'Vandalism']  # Update this list as needed

# Path to the directory containing your *original* videos (e.g., "../ucf_crime_full/").
SOURCE_VIDEO_DIR = Path("../ucf_crime_full")

# Path to your CSV file.
CSV_FILE_PATH = Path("./clips.csv")

# The target duration for your sub-clips (in seconds).
TARGET_CLIP_DURATION_S = 10.0

# The step size (stride) for creating overlapping clips (in seconds).
OVERLAP_STEP_SIZE_S = 2.5

# --------------------------

def to_seconds(time_str: str) -> float:
    """Converts a 'HH:MM:SS' or 'HH:MM:SS.ms' string to seconds."""
    try:
        if '.' in time_str:
            parts = time_str.split('.')
            main_time = parts[0]
            millis = float(f"0.{parts[1]}")
        else:
            main_time = time_str
            millis = 0.0
            
        h, m, s = map(int, main_time.split(':'))
        return (h * 3600) + (m * 60) + s + millis
    except ValueError as e:
        print(f"Error converting time string '{time_str}': {e}. Returning 0.0")
        return 0.0

def count_clips_for_segment(start_s: float, end_s: float) -> int:
    """
    Counts how many clips would be created for a given time segment.
    Clips can extend beyond end_s, and are created as long as the start time <= end_s.
    For segments <= 10s, uses 1s step; for longer segments, uses 2.5s step.
    """
    duration_s = end_s - start_s
    if duration_s <= 0:
        return 0
    
    # Determine step size based on duration
    if duration_s <= TARGET_CLIP_DURATION_S:
        step_size = 1.0  # 1 second step for short clips
    else:
        step_size = OVERLAP_STEP_SIZE_S  # 2.5 seconds for longer clips
    
    num_clips = 0
    current_start_s = start_s
    
    # Create clips as long as the start time is <= end_s
    while current_start_s <= end_s:
        num_clips += 1
        current_start_s += step_size

    return num_clips

def process_video_clips():
    """
    Main function to read the CSV and count video clips per class.
    """
    
    # --- 1. Load CSV File ---
    if not CSV_FILE_PATH.exists():
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please update the 'CSV_FILE_PATH' variable in the script.")
        return

    if not SOURCE_VIDEO_DIR.exists() or not SOURCE_VIDEO_DIR.is_dir():
        print(f"Error: Source video directory not found at {SOURCE_VIDEO_DIR}")
        print("Please update the 'SOURCE_VIDEO_DIR' variable in the script.")
        return

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check for required columns
    required_cols = ['exit', 'Start Time', 'End Time']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain columns: {required_cols}")
        return
        
    print(f"Loaded {len(df)} clip entries from {CSV_FILE_PATH.name}")
    print(f"Considering classes: {CLASSES_TO_CONSIDER}")
    print("--- Starting processing ---")
    
    start_time = time.time()

    # Dictionaries to track counts
    original_videos_per_class = defaultdict(set)  # Use set to track unique video stems
    clips_per_class = defaultdict(int)

    # --- 2. Process Each Row ---
    for idx, row in df.iterrows():
        try:
            video_path_relative = Path(row['exit'])
            start_time_str = row['Start Time']
            end_time_str = row['End Time']

            # Get class name from the path
            class_name = str(video_path_relative.parent)
            
            # Skip if class not in considered classes
            if class_name not in CLASSES_TO_CONSIDER:
                continue

            # Get video file stem
            video_stem = video_path_relative.stem

            # Add to original videos set
            original_videos_per_class[class_name].add(video_stem)

            start_s = to_seconds(start_time_str)
            end_s = to_seconds(end_time_str)
            duration_s = end_s - start_s

            if duration_s <= 0:
                print(f"[Warning] Invalid time range for {video_path_relative} (Row {idx}): {start_time_str} to {end_time_str}. Skipping.")
                continue

            # Count clips for this segment
            num_clips = count_clips_for_segment(start_s, end_s)
            clips_per_class[class_name] += num_clips

        except Exception as e:
            print(f"[ERROR] Unhandled error on row {idx}: {e}")
            continue
            
    end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    
    # Print results
    print("\n--- Results ---")
    for class_name in sorted(original_videos_per_class.keys()):
        num_original = len(original_videos_per_class[class_name])
        num_clips = clips_per_class[class_name]
        print(f"Class '{class_name}': {num_original} original videos, {num_clips} clips")


if __name__ == "__main__":
    # --- Dependencies ---
    # This script requires 'pandas'.
    # You can install it by running:
    # pip install pandas
    # --------------------
    
    process_video_clips()