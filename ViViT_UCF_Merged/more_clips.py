import pandas as pd
from pathlib import Path
from moviepy.editor import VideoFileClip
import time
import math

# --- User Configuration ---
# PLEASE UPDATE THESE PATHS BEFORE RUNNING

# 1. Path to the directory containing your *original* videos
SOURCE_VIDEO_DIR = Path("../ucf_crime_full")

# 2. Path to the new directory where you want to save the *augmented clips*.
OUTPUT_DATASET_DIR = Path("../clips_ucfcrime")

# 3. Path to your CSV file.
CSV_FILE_PATH = Path("clips.csv")

# --- New Augmentation Settings ---

# 4. Specify the *only* class you want to process from the CSV.
#    (e.g., "Abuse", "Arrest", "Vandalism")
TARGET_CLASS = "RoadAccidents" 

# 5. The exact length (in seconds) of each *new* clip to be created.
#    This will be the standard (and maximum) length.
MAX_CLIP_LENGTH_S = 10.0

# 6. How to calculate the step size 'd' for augmentation.
#    This is the "d" you described. We'll set it to 15% of the total
#    CSV time range.
#    - For a 20s range (e.g., 0:10 to 0:30), d = 20 * 0.15 = 3 seconds.
#    - For a 3min range (180s), d = 180 * 0.15 = 27 seconds.
AUGMENT_STEP_PERCENT = 0.15  # (15%)

# 7. Clamps for the dynamic step 'd' to keep it reasonable.
MIN_STEP_S = 2.0    # 'd' will never be smaller than 2s
MAX_STEP_S = 40.0   # 'd' will never be larger than 40s

# ---------------------------------

def to_seconds(time_str: str) -> float:
    """Converts 'HH:MM:SS' or 'MM:SS' string to seconds."""
    try:
        parts = list(map(float, time_str.split(':')))
        if len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        elif len(parts) == 1: # Just seconds
             return parts[0]
        else:
            raise ValueError(f"Invalid time format: {time_str}")
    except (ValueError, TypeError, AttributeError) as e:
        print(f"[Error] Could not parse time '{time_str}': {e}")
        return 0.0

def trim_video(source_path: Path, output_path: Path, start_s: float, end_s: float):
    """
    Trims a single video clip using moviepy and saves it.
    Uses default 'libx264' codec.
    """
    try:
        print(f"    Creating clip: {output_path.name} [Time: {start_s:.2f}s to {end_s:.2f}s]")
        
        # Load the source video, create the subclip, and write it.
        with VideoFileClip(str(source_path)) as video:
            # Ensure clip is within video bounds
            clip_start = max(0, start_s)
            clip_end = min(video.duration, end_s)
            
            if clip_start >= clip_end:
                print(f"    [Skipping] Clip start time ({clip_start}) is after end time ({clip_end}).")
                return

            clip = video.subclip(clip_start, clip_end)
            # We let moviepy pick the default codec (libx264)
            clip.write_videofile(str(output_path), logger=None, audio_codec="aac")
            
    except Exception as e:
        print(f"    [ERROR] Failed to create clip {output_path.name}: {e}")
        pass

def get_video_path_and_class(csv_entry: str) -> (Path, str, str):
    """
    Splits a CSV entry like "Abuse/Abuse001_x264.mp4" into parts.
    Returns: (Path('Abuse/Abuse001_x264.mp4'), 'Abuse', 'Abuse001_x264')
    """
    try:
        entry_path = Path(csv_entry)
        class_name = entry_path.parts[0]  # 'Abuse'
        video_stem = entry_path.stem      # 'Abuse001_x264'
        return entry_path, class_name, video_stem
    except Exception as e:
        print(f"[Error] Could not parse video entry '{csv_entry}': {e}")
        return None, None, None

def process_video_row(row_index: int, row: pd.Series, source_dir: Path, output_dir: Path):
    """
    Processes a single row from the CSV file and generates augmented clips.
    """
    csv_entry = row['exit']
    start_time_str = row['Start Time']
    end_time_str = row['End Time']
    
    # --- 1. Get video paths and class ---
    relative_path, class_name, video_stem = get_video_path_and_class(csv_entry)
    if not relative_path:
        return

    # Check if this row is for the target class
    if class_name != TARGET_CLASS:
        return # Skip this row silently

    source_video_path = source_dir / relative_path
    output_class_dir = output_dir / class_name
    
    if not source_video_path.exists():
        print(f"  [Skipping] Source video not found: {source_video_path}")
        return

    # Create the output directory (e.g., .../NewDataset/Abuse/)
    output_class_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Parse times and duration ---
    start_s = to_seconds(start_time_str)
    end_s = to_seconds(end_time_str)
    range_duration_s = end_s - start_s

    if range_duration_s <= 0:
        print(f"  [Skipping] Invalid time range for {video_stem} (Start: {start_s}s, End: {end_s}s)")
        return

    # --- 3. Handle different clip types ---
    print(f"  Processing {video_stem} [Row {row_index}]: Range {start_s:.2f}s to {end_s:.2f}s")
    
    # Calculate the dynamic step size 'd' based on the CSV range
    dynamic_step = range_duration_s * AUGMENT_STEP_PERCENT
    step_size_d = max(MIN_STEP_S, min(dynamic_step, MAX_STEP_S))
    
    clip_counter = 0

    # --- 4. NEW LOGIC ---

    # Check if the CSV range is *shorter* than or EQUAL TO our max clip length
    if range_duration_s <= MAX_CLIP_LENGTH_S:
        print(f"    Range ({range_duration_s:.2f}s) is <= max clip length ({MAX_CLIP_LENGTH_S}s).")
        
        # Only save the "orig" clip if it's *actually* shorter
        if range_duration_s < MAX_CLIP_LENGTH_S:
            print(f"    Saving the original short clip first.")
            # 1. Save the original short clip with a special name
            out_name = f"{video_stem}_mul_clip_{row_index}_orig.mp4"
            out_path = output_class_dir / out_name
            trim_video(source_video_path, out_path, start_s, end_s)
        
        # 2. Now, loop to create 10s clips anchored in this range
        print(f"    Augmenting with {MAX_CLIP_LENGTH_S}s clips starting from {start_s:.2f}s...")
        
        current_start_s = start_s
        
        # Loop as long as the *start time* of the clip is within the CSV range
        while current_start_s <= end_s:
            clip_end_s = current_start_s + MAX_CLIP_LENGTH_S
            
            out_name = f"{video_stem}_mul_clip_{row_index}_{clip_counter}.mp4"
            out_path = output_class_dir / out_name
            
            trim_video(source_video_path, out_path, current_start_s, clip_end_s)
            
            # Move to the next start time
            current_start_s += step_size_d
            clip_counter += 1

            # Safety break in case d is near zero
            if step_size_d < 0.1:
                break
        
        print(f"    -> Generated 1 original clip and {clip_counter} augmented {MAX_CLIP_LENGTH_S}s clips with a step of {step_size_d:.2f}s")

    else:
        # This is the *original* logic, for when range_duration_s >= MAX_CLIP_LENGTH_S
        # Loop and create clips, stopping when the *end* of the clip would go past 'end_s'
        print(f"    Range ({range_duration_s:.2f}s) is >= max length. Creating sliding {MAX_CLIP_LENGTH_S}s clips...")
        
        current_start_s = start_s
        
        while current_start_s + MAX_CLIP_LENGTH_S <= end_s:
            clip_end_s = current_start_s + MAX_CLIP_LENGTH_S
            
            # Use new naming convention: {video_stem}_mul_clip_{ROW_INDEX}_{CLIP_NUMBER}.mp4
            out_name = f"{video_stem}_mul_clip_{row_index}_{clip_counter}.mp4"
            out_path = output_class_dir / out_name
            
            trim_video(source_video_path, out_path, current_start_s, clip_end_s)
            
            # Move to the next start time
            current_start_s += step_size_d
            clip_counter += 1

        print(f"    -> Generated {clip_counter} augmented clips with a step of {step_size_d:.2f}s")


def main():
    """
    Main function to read CSV and process all videos.
    """
    # --- 1. Validate Configuration ---
    if not SOURCE_VIDEO_DIR.exists() or not SOURCE_VIDEO_DIR.is_dir():
        print(f"Error: Source video directory not found at {SOURCE_VIDEO_DIR}")
        print("Please update the 'SOURCE_VIDEO_DIR' variable in the script.")
        return

    if not CSV_FILE_PATH.exists():
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please update the 'CSV_FILE_PATH' variable in the script.")
        return
        
    if not TARGET_CLASS:
        print(f"Error: 'TARGET_CLASS' is not set.")
        print("Please update this variable to the class name you want to process (e.g., 'Abuse').")
        return

    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Starting Video Augmentation Script ---")
    print(f"Source videos:    {SOURCE_VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DATASET_DIR}")
    print(f"CSV file:         {CSV_FILE_PATH}")
    print(f"Target Class:     '{TARGET_CLASS}'")
    print(f"Target Clip Size: {MAX_CLIP_LENGTH_S}s")
    print("------------------------------------------")
    
    start_time = time.time()

    # --- 2. Read CSV ---
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        # Clean up column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'exit' not in df.columns or 'Start Time' not in df.columns or 'End Time' not in df.columns:
        print(f"Error: CSV must contain 'exit', 'Start Time', and 'End Time' columns.")
        print(f"Found columns: {list(df.columns)}")
        return
        
    # --- 3. Filter DataFrame for Target Class ---
    # This is much faster than checking every row
    df_filtered = df[df['exit'].str.startswith(f"{TARGET_CLASS}/", na=False)].copy()
    
    if df_filtered.empty:
        print(f"No entries found in CSV for the target class '{TARGET_CLASS}'.")
        print("Please check your 'TARGET_CLASS' variable and CSV file.")
        return
        
    print(f"Found {len(df_filtered)} CSV entries for class '{TARGET_CLASS}'. Processing...")

    # --- 4. Process each video row ---
    for index, row in df_filtered.iterrows():
        print(f"\nProcessing CSV Row {index}...")
        try:
            process_video_row(index, row, SOURCE_VIDEO_DIR, OUTPUT_DATASET_DIR)
        except Exception as e:
            # Catch any unexpected errors on a per-row basis
            print(f"[CRITICAL ERROR] Failed to process row {index} ({row.get('exit', 'N/A')}).")
            print(f"Error: {e}")
            continue

    end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # --- Dependencies ---
    # This script requires 'pandas' and 'moviepy'.
    # You can install them by running:
    # pip install pandas moviepy
    #
    # You also need 'ffmpeg' installed on your system.
    # --------------------
    
    main()


