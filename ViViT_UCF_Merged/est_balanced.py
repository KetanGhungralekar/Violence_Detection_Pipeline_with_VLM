import re
from pathlib import Path
from collections import defaultdict
import math

# --- CONFIGURATION ---
CLIPS_DIR = Path("../extracted_clips_final")

# The target range of clips per class
TARGET_MIN = 1300
TARGET_MAX = 1350
CLIP_DURATION = 10.0

# We will try step sizes between 0.1s (very dense) and 10.0s (no overlap)
MIN_STEP_LIMIT = 0.1
MAX_STEP_LIMIT = 10.0
SEARCH_PRECISION = 0.1  # We will adjust step size in 0.1s increments

TARGET_CLASSES = [
    'Explosion', 'Abuse', 'Shooting', 'Arrest', 'Arson', 
    'RoadAccidents', 'Fighting', 'Vandalism', 'Robbery', 'Assault'
]

# ---------------------

def parse_clip_filename(filename):
    """Parses start and end times from filename."""
    pattern = r"(.+?_x264).*_start_([\d.]+)s_end_([\d.]+)s"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), float(match.group(2)), float(match.group(3))
    return None, 0.0, 0.0

def calculate_counts(islands, step_size):
    """Calculates total clips possible for a set of time islands given a specific step."""
    if step_size <= 0: return 0
    total_clips = 0
    for start, end in islands:
        duration = end - start
        if duration < CLIP_DURATION:
            continue
        # Logic: how many steps fit in the valid sliding window?
        # Sliding window size = Duration - Clip_Length
        count = int((duration - CLIP_DURATION) / step_size) + 1
        total_clips += count
    return total_clips

def find_ideal_step(islands, current_count):
    """
    Iteratively finds the step size that results in a count closest to the target.
    """
    # 1. Calculate Max Possible (Densest overlap)
    max_possible = calculate_counts(islands, MIN_STEP_LIMIT)
    
    # If we can't even reach the target with 0.1s overlap, return max effort
    if max_possible < TARGET_MIN:
        return MIN_STEP_LIMIT, max_possible, "Maxed Out"

    # 2. If we have too many, or just enough, search for the 'Goldilocks' step
    best_step = 1.0
    best_count = 0
    closest_diff = float('inf')

    # Scan from 0.1 to 10.0 to find the step that lands us in the 1300-1350 range
    # This is a brute force search, but very fast for this scale
    step = MIN_STEP_LIMIT
    while step <= MAX_STEP_LIMIT:
        count = calculate_counts(islands, step)
        
        # If we hit the target window, stop immediately
        if TARGET_MIN <= count <= TARGET_MAX:
            return step, count, "Balanced"
        
        # Keep track of the closest we got if we can't hit the exact window
        # We prefer being slightly over target than under
        if count > TARGET_MIN:
             diff = abs(count - ((TARGET_MIN + TARGET_MAX)/2))
             if diff < closest_diff:
                 closest_diff = diff
                 best_step = step
                 best_count = count
        
        step += SEARCH_PRECISION

    return best_step, best_count, "Optimized"

def main():
    print(f"--- Balancing Dataset to Target: {TARGET_MIN}-{TARGET_MAX} clips ---")
    
    # 1. Load Data
    video_data = defaultdict(lambda: defaultdict(list))
    for class_name in TARGET_CLASSES:
        class_dir = CLIPS_DIR / class_name
        if not class_dir.exists(): continue
        for f in class_dir.glob("*.mp4"):
            stem, s, e = parse_clip_filename(f.name)
            if stem: video_data[class_name][stem].append((s, e))

    print(f"{'CLASS':<15} | {'ORIGINAL':<8} | {'CALC. STEP':<10} | {'EST. FINAL':<10} | {'STATUS':<10}")
    print("-" * 65)

    grand_total = 0

    for class_name in TARGET_CLASSES:
        all_islands = []
        original_clip_count = 0

        # 2. Merge clips into islands per video
        for stem, clips in video_data[class_name].items():
            original_clip_count += len(clips)
            clips.sort(key=lambda x: x[0])
            
            # Merging logic
            if not clips: continue
            curr_s, curr_e = clips[0]
            for next_s, next_e in clips[1:]:
                if next_s <= curr_e: # Overlap or continuous
                    curr_e = max(curr_e, next_e)
                else:
                    all_islands.append((curr_s, curr_e))
                    curr_s, curr_e = next_s, next_e
            all_islands.append((curr_s, curr_e))

        # 3. Find the perfect step size
        # We do max(len, ...) to ensure we don't estimate fewer than we currently have
        # (Though usually the re-estimation covers existing clips)
        rec_step, est_count, status = find_ideal_step(all_islands, original_clip_count)
        
        # Safety check: if the calculation yields fewer than we have, just report what we have
        if est_count < original_clip_count:
            est_count = original_clip_count
            status = "Kept Orig"
            rec_step = "N/A"
        else:
            rec_step = f"{rec_step:.1f}s"

        grand_total += est_count
        print(f"{class_name:<15} | {original_clip_count:<8} | {rec_step:<10} | {est_count:<10} | {status:<10}")

    print("-" * 65)
    print(f"TOTAL DATASET SIZE: {grand_total}")

if __name__ == "__main__":
    main()