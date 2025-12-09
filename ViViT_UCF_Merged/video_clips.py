import pandas as pd
from pathlib import Path
from moviepy.editor import VideoFileClip
import time
import math

# --- User Configuration ---
# PLEASE UPDATE THESE PATHS BEFORE RUNNING

# Path to the directory containing your *original* videos (e.g., "MyVideos/").
# The script will look for videos relative to this path (e.g., SOURCE_VIDEO_DIR / "Abuse/Abuse001_x264.mp4")
SOURCE_VIDEO_DIR = Path("path/to/your/original_videos_folder")

# Path to the new directory where you want to save the *trimmed clips*.
# This directory will be created if it doesn't exist.
OUTPUT_DATASET_DIR = Path("path/to/your/new_dataset_folder")

# Path to your CSV file.
CSV_FILE_PATH = Path("crime_videos_for_splitting (1).xlsx - Sheet1.csv")

# The target duration for your sub-clips (in seconds).
TARGET_CLIP_DURATION_S = 10.0

# The step size (stride) for creating overlapping clips (in seconds).
# A 2.5s step on a 10s clip creates 7.5s of overlap.
# You mentioned a "5-7 second buffer"; this step size controls that.
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

def trim_video(source_path: Path, output_path: Path, start_s: float, end_s: float):
    """
    Trims a video clip using moviepy and saves it.
    """
    try:
        # Check if start and end times are valid
        if start_s >= end_s:
            print(f"  [Skipping] Start time ({start_s}s) is after or at end time ({end_s}s) for {output_path.name}")
            return
            
        print(f"  Creating clip: {output_path.name} [Time: {start_s:.2f}s to {end_s:.2f}s]")
        
        # Load the source video, create the subclip, and write it to the output file.
        # 'logger=None' prevents moviepy from printing too much to the console.
        # We specify codecs to ensure compatibility.
        with VideoFileClip(str(source_path)) as video:
            # Ensure subclip times are within the video's duration
            video_duration = video.duration
            if start_s > video_duration:
                print(f"  [Skipping] Start time ({start_s}s) is beyond video duration ({video_duration}s)")
                return
                
            # Clamp the end time to the video's actual duration
            safe_end_s = min(end_s, video_duration)
            
            if start_s >= safe_end_s:
                print(f"  [Skipping] Adjusted start time ({start_s}s) is after or at adjusted end time ({safe_end_s}s)")
                return

            clip = video.subclip(start_s, safe_end_s)
            clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)
            
    except Exception as e:
        print(f"  [ERROR] Failed to create clip {output_path.name}: {e}")
        # This can happen if the video file is corrupt or times are invalid
        pass

def process_video_clips():
    """
    Main function to read the CSV and process all video clips.
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
    print(f"Source videos will be read from: {SOURCE_VIDEO_DIR}")
    print(f"Output clips will be saved to: {OUTPUT_DATASET_DIR}")
    print("--- Starting processing ---")
    
    start_time = time.time()

    # --- 2. Process Each Row ---
    for idx, row in df.iterrows():
        try:
            video_path_relative = Path(row['exit'])
            start_time_str = row['Start Time']
            end_time_str = row['End Time']

            # --- 3. Prepare Paths and Times ---
            source_video_full_path = SOURCE_VIDEO_DIR / video_path_relative
            
            # Get class name (e.g., "Abuse") from the path
            class_name = video_path_relative.parent
            # Get video file stem (e.g., "Abuse001_x264")
            video_stem = video_path_relative.stem

            # Create the output directory for the class (e.g., .../new_dataset_folder/Abuse/)
            output_class_dir = OUTPUT_DATASET_DIR / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            if not source_video_full_path.exists():
                print(f"[Warning] Source video not found, skipping: {source_video_full_path}")
                continue

            start_s = to_seconds(start_time_str)
            end_s = to_seconds(end_time_str)
            duration_s = end_s - start_s

            if duration_s <= 0:
                print(f"[Warning] Invalid time range for {video_path_relative} (Row {idx}): {start_time_str} to {end_time_str}. Skipping.")
                continue

            print(f"\nProcessing Row {idx}: {video_path_relative} ({duration_s:.2f}s clip)")

            # --- 4. Main Clipping Logic ---
            
            # We use the DataFrame index (idx) to make clip names unique for each *row*
            # and a sub_clip_counter (sc) for clips generated *within* a row.
            
            if duration_s <= TARGET_CLIP_DURATION_S:
                # Clip is short or equal to target, just create one clip
                out_name = f"{video_stem}_clip_{idx}_0.mp4"
                out_path = output_class_dir / out_name
                trim_video(source_video_full_path, out_path, start_s, end_s)
                
            else:
                # Clip is longer than target, create overlapping sub-clips
                sub_clip_counter = 0
                current_start_s = start_s
                
                # Loop and create clips moving forward
                while (current_start_s + TARGET_CLIP_DURATION_S) <= end_s:
                    current_end_s = current_start_s + TARGET_CLIP_DURATION_S
                    out_name = f"{video_stem}_clip_{idx}_{sub_clip_counter}.mp4"
                    out_path = output_class_dir / out_name
                    
                    trim_video(source_video_full_path, out_path, current_start_s, current_end_s)
                    
                    sub_clip_counter += 1
                    current_start_s += OVERLAP_STEP_SIZE_S

                # After the loop, check if we need to add the *final* 10-second segment
                # This ensures the very end of the specified clip is captured
                final_clip_start_s = end_s - TARGET_CLIP_DURATION_S
                
                # Check if this final clip start is *after* the *start* of the last clip we created
                # (current_start_s is already incremented, so we check against the previous step)
                last_loop_start_s = current_start_s - OVERLAP_STEP_SIZE_S
                
                if final_clip_start_s > last_loop_start_s:
                    out_name = f"{video_stem}_clip_{idx}_{sub_clip_counter}.mp4"
                    out_path = output_class_dir / out_name
                    trim_video(source_video_full_path, out_path, final_clip_start_s, end_s)

        except Exception as e:
            print(f"[ERROR] Unhandled error on row {idx}: {e}")
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
    # If moviepy can't find it, it will tell you.
    # Easiest way to get it:
    # - Windows: Download from https://ffmpeg.org/download.html
    # - macOS (using Homebrew): brew install ffmpeg
    # - Linux (using apt): sudo apt install ffmpeg
    # --------------------
    
    process_video_clips()
