import os
import re
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---

# Paths
# The directory containing the clips you ALREADY extracted
CLIPS_DIR = Path("../extracted_clips_final")

# The directory of original videos (kept for reference/path correctness, though not strictly opened in this estimation)
ORIGINAL_VIDEOS_DIR = Path("../ucf_crime_full")

# Classes to process
TARGET_CLASSES = [
    'Explosion', 'Abuse', 'Shooting', 'Arrest', 'Arson', 
    'RoadAccidents', 'Fighting', 'Vandalism', 'Robbery', 'Assault'
]

# The target clip length (should remain 10.0 based on your requirements)
CLIP_DURATION = 10.0

# AUGMENTATION STRIDE (Seconds)
# This determines how "dense" the new clips will be.
# 1.0 means you generate a clip at 1.0s, 2.0s, 3.0s, etc. within the valid range.
# Smaller number = More videos. (e.g., 0.5 or 1.0 are common for augmentation)
NEW_STEP_SIZE = 1.0 

# ---------------------

def parse_clip_filename(filename):
    """
    Parses filenames like:
    Assault052_x264_clip_436_0003_start_12.50s_end_22.50s.mp4
    Returns: (video_stem, start_time, end_time)
    """
    # Regex to capture the original video stem (up to _x264) and the times
    # Matches: (Assault052_x264) ... start_(12.50)s ... end_(22.50)s
    pattern = r"(.+?_x264).*_start_([\d.]+)s_end_([\d.]+)s"
    match = re.search(pattern, filename)
    
    if match:
        stem = match.group(1)
        start = float(match.group(2))
        end = float(match.group(3))
        return stem, start, end
    return None, 0.0, 0.0

def estimate_clips_in_range(start_time, end_time, step_size, duration):
    """
    Calculates how many fixed-duration clips fit in a time range 
    using a specific step size.
    """
    valid_duration = end_time - start_time
    
    if valid_duration < duration:
        return 0
    
    # Formula: floor((TotalDuration - ClipDuration) / Step) + 1
    count = int((valid_duration - duration) // step_size) + 1
    return max(0, count)

def main():
    if not CLIPS_DIR.exists():
        print(f"Error: Directory not found: {CLIPS_DIR}")
        return

    print(f"--- Scanning {CLIPS_DIR} ---")
    print(f"--- Estimating with New Step Size: {NEW_STEP_SIZE}s ---")
    
    # Data structure: data[class][video_stem] = list of (start, end) tuples
    video_data = defaultdict(lambda: defaultdict(list))
    
    # 1. Scan and Group Files
    total_files_found = 0
    
    for class_name in TARGET_CLASSES:
        class_dir = CLIPS_DIR / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory missing: {class_dir}")
            continue
            
        for file_path in class_dir.glob("*.mp4"):
            stem, start, end = parse_clip_filename(file_path.name)
            if stem:
                video_data[class_name][stem].append((start, end))
                total_files_found += 1

    print(f"Found {total_files_found} existing clips across target classes.\n")
    
    print(f"{'CLASS':<15} | {'EXISTING':<10} | {'EST. NEW':<10} | {'EST. TOTAL':<10}")
    print("-" * 55)

    # 2. Calculate Estimates
    grand_total_existing = 0
    grand_total_new = 0

    for class_name in TARGET_CLASSES:
        existing_count_class = 0
        estimated_total_class = 0
        
        # Process each original video source within the class
        for stem, clips in video_data[class_name].items():
            # Sort clips by start time
            clips.sort(key=lambda x: x[0])
            
            existing_count_class += len(clips)
            
            if not clips:
                continue

            # --- Logic: Find Continuous Islands ---
            # We merge overlapping clips to find the full range of valid video data available
            # e.g., [0-10], [5-15] becomes a continuous block [0-15]
            
            merged_blocks = []
            if clips:
                current_start, current_end = clips[0]
                
                for i in range(1, len(clips)):
                    next_start, next_end = clips[i]
                    
                    # If the next clip starts before (or exactly when) the current one ends
                    # It is part of the same continuous block
                    if next_start <= current_end:
                        current_end = max(current_end, next_end)
                    else:
                        # Gap detected, push current block and start new one
                        merged_blocks.append((current_start, current_end))
                        current_start, current_end = next_start, next_end
                
                merged_blocks.append((current_start, current_end))
            
            # --- Logic: Resample the Islands ---
            # Calculate max possible 10s clips in these blocks with the new step size
            possible_clips_for_video = 0
            for b_start, b_end in merged_blocks:
                possible_clips_for_video += estimate_clips_in_range(
                    b_start, b_end, NEW_STEP_SIZE, CLIP_DURATION
                )
            
            # We take the MAX of (existing, possible). 
            # We never want the estimation to result in fewer clips than we already have 
            # (which could happen if the new grid doesn't align with old specific timestamps).
            estimated_total_class += max(len(clips), possible_clips_for_video)

        # Calculate "New" for this class
        new_clips_class = estimated_total_class - existing_count_class
        
        grand_total_existing += existing_count_class
        grand_total_new += new_clips_class
        
        print(f"{class_name:<15} | {existing_count_class:<10} | +{new_clips_class:<9} | {estimated_total_class:<10}")

    print("-" * 55)
    print(f"{'TOTAL':<15} | {grand_total_existing:<10} | +{grand_total_new:<9} | {grand_total_existing + grand_total_new:<10}")

if __name__ == "__main__":
    main()