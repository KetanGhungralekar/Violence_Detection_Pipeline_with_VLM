import os
import re
import time
from pathlib import Path
from collections import defaultdict
from moviepy.editor import VideoFileClip

# --- CONFIGURATION ---

EXTRACTED_DIR = Path("../extracted_clips_final")
SOURCE_VIDEO_DIR = Path("../ucf_crime_full")
TARGET_DURATION = 10.0

# 1. Step Sizes (From your calculation)
CLASS_STEP_CONFIG = {
    'Explosion': 1.8,
    'Abuse': 1.9,
    'Shooting': 0.3,
    'Arrest': 5.1,
    'Arson': 3.0,
    'RoadAccidents': 0.9,
    'Fighting': 0.8,
    'Vandalism': 0.5,
    'Robbery': 4.7,
    'Assault': 1.7,
}

# 2. Hard Limits (From your "Balanced" table)
# The script will STOP creating clips once the folder contains this many.
TARGET_LIMITS = {
    'Explosion': 1340,
    'Abuse': 1340,
    'Shooting': 1340,
    'Arrest': 1340,
    'Arson': 1340,
    'RoadAccidents': 1340,
    'Fighting': 1340,
    'Vandalism': 1340,
    'Robbery': 1340,
    'Assault': 1340,
}

# ---------------------

def parse_clip_info(filename):
    pattern = r"(.+?_x264)_clip_(\d+)_.*_start_([\d.]+)s_end_([\d.]+)s"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2), float(match.group(3)), float(match.group(4))
    return None, None, 0.0, 0.0

def get_source_video_path(class_name, video_stem):
    path = SOURCE_VIDEO_DIR / class_name / f"{video_stem}.mp4"
    if path.exists(): return path
    path = SOURCE_VIDEO_DIR / class_name / f"{video_stem}.avi"
    if path.exists(): return path
    return None

def process_class_augmentation(class_name, step_size):
    class_dir = EXTRACTED_DIR / class_name
    target_count = TARGET_LIMITS.get(class_name, 1350)
    
    if not class_dir.exists():
        print(f"Skipping {class_name} (Directory not found)")
        return

    # 1. Count Existing Files
    files = list(class_dir.glob("*.mp4"))
    current_count = len(files)
    
    print(f"--- Processing {class_name} ---")
    print(f"    Target: {target_count} | Current: {current_count} | Step: {step_size}s")

    if current_count >= target_count:
        print(f"    [Check] Target reached or exceeded. Skipping.")
        return

    # 2. Map existing clips
    video_data = defaultdict(list)
    existing_starts = defaultdict(set)

    for file_path in files:
        stem, row_idx, start, end = parse_clip_info(file_path.name)
        if stem:
            video_data[stem].append((start, end, row_idx))
            existing_starts[stem].add(round(start, 2))

    new_clips_created = 0

    # 3. Process stems
    # We sort stems to ensure deterministic processing, but you could shuffle if you want random sampling
    sorted_stems = sorted(video_data.keys())

    for stem in sorted_stems:
        # CHECK LIMIT: Stop immediately if we hit the target
        if (current_count + new_clips_created) >= target_count:
            print(f"    [Limit Reached] Stopping augmentation for {class_name}.")
            break

        clips = video_data[stem]
        clips.sort(key=lambda x: x[0])
        
        source_path = get_source_video_path(class_name, stem)
        if not source_path: continue

        # Create Islands
        islands = []
        if clips:
            curr_s, curr_e, curr_row = clips[0]
            for next_s, next_e, next_row in clips[1:]:
                if next_s <= curr_e: 
                    curr_e = max(curr_e, next_e)
                else:
                    islands.append((curr_s, curr_e, curr_row))
                    curr_s, curr_e, curr_row = next_s, next_e, next_row
            islands.append((curr_s, curr_e, curr_row))

        # Generate NEW clips
        try:
            with VideoFileClip(str(source_path)) as video:
                video_duration = video.duration

                for (island_start, island_end, row_idx) in islands:
                    
                    # CHECK LIMIT inside the loop too
                    if (current_count + new_clips_created) >= target_count:
                        break

                    scan_time = island_start
                    
                    while scan_time + TARGET_DURATION <= island_end + 0.01:
                        # CHECK LIMIT inside the inner loop
                        if (current_count + new_clips_created) >= target_count:
                            break

                        # Duplicate Check
                        is_duplicate = False
                        for exist_s in existing_starts[stem]:
                            if abs(exist_s - scan_time) < 0.1: 
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            new_start = scan_time
                            new_end = scan_time + TARGET_DURATION
                            
                            if new_end > video_duration: break

                            clip_name = (
                                f"{stem}_clip_{row_idx}_aug_{int(new_start*100):04d}_"
                                f"start_{new_start:.2f}s_end_{new_end:.2f}s_augment.mp4"
                            )
                            output_path = class_dir / clip_name
                            
                            new_clip = video.subclip(new_start, new_end)
                            new_clip.write_videofile(
                                str(output_path), 
                                codec="libx264", 
                                audio=False, 
                                verbose=False, 
                                logger=None
                            )
                            
                            new_clips_created += 1
                            
                        scan_time += step_size

        except Exception as e:
            print(f"    [Error] {stem}: {e}")
            continue

    print(f"    -> Added {new_clips_created} clips. Final Total: {current_count + new_clips_created}")

def main():
    if not EXTRACTED_DIR.exists() or not SOURCE_VIDEO_DIR.exists():
        print("Error: Check directory paths.")
        return

    print("--- Starting Controlled Augmentation ---")
    start_time = time.time()

    for class_name, step_size in CLASS_STEP_CONFIG.items():
        process_class_augmentation(class_name, step_size)

    print(f"\n--- Done! Total time: {time.time() - start_time:.2f}s ---")

if __name__ == "__main__":
    main()