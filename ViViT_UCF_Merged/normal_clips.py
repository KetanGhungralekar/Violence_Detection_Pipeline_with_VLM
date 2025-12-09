import time
import math
import re
from pathlib import Path
from moviepy.editor import VideoFileClip

# --- User Configuration ---
# PLEASE UPDATE THESE PATHS BEFORE RUNNING

# Path to the directory containing your *original* videos for this class
# (e.g., "MyVideos/Normal_Events/").
# The script will search this folder *and all its subfolders* for videos.
SOURCE_VIDEO_DIR = Path("../ucf_crime_full/Normal_Videos_for_Event_Recognition")

# Path to the new directory where you want to save the *trimmed clips*.
# This directory will be created if it doesn't exist.
OUTPUT_CLIP_DIR = Path("../clips_ucfcrime/Normal_Videos_for_Event_Recognition")

# The exact duration (in seconds) for each disjoint clip.
# You mentioned 10-15; I've set the default to 10.
# You can change this to 12, 15, or any other value you prefer.
CLIP_DURATION_S = 10.0

# Minimum duration (in seconds) for a clip to be saved.
# Any remaining chunk shorter than this will be ignored.
MIN_CLIP_DURATION_S = 1.0

# List of video file extensions to look for.
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

# List of video numbers to EXCLUDE from processing
# Add the numbers from your list here (e.g., [109, 110, 197, 307])
EXCLUDE_VIDEO_NUMBERS = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 16, 17, 20, 21, 22, 23, 26, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 49, 52, 53, 54, 55, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 201, 202, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 249, 250, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307]  # UPDATE THIS LIST with numbers to skip
# --------------------------

def should_skip_video(video_filename: str, exclude_numbers: list) -> bool:
    """
    Check if a video should be skipped based on its number.
    Matches patterns like Normal_Videos109 or Normal_Videos_001
    """
    match = re.search(r'Normal_Videos_?(\d+)', video_filename)
    if match:
        video_number = int(match.group(1))
        if video_number in exclude_numbers:
            return True
    return False

def trim_video_disjoint(source_path: Path, output_path: Path, start_s: float, end_s: float):
    """
    Trims a single video clip using moviepy and saves it.
    """
    try:
        print(f"  Creating clip: {output_path.name} [Time: {start_s:.2f}s to {end_s:.2f}s]")
        
        # Load the source video, create the subclip, and write it.
        with VideoFileClip(str(source_path)) as video:
            clip = video.subclip(start_s, end_s)
            clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)
            
    except Exception as e:
        print(f"  [ERROR] Failed to create clip {output_path.name}: {e}")
        pass

def process_disjoint_clips():
    """
    Main function to find all videos and cut them into disjoint clips.
    """
    
    # --- 1. Setup Paths ---
    if not SOURCE_VIDEO_DIR.exists() or not SOURCE_VIDEO_DIR.is_dir():
        print(f"Error: Source video directory not found at {SOURCE_VIDEO_DIR}")
        print("Please update the 'SOURCE_VIDEO_DIR' variable in the script.")
        return

    # Create the output directory
    OUTPUT_CLIP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source videos will be read from: {SOURCE_VIDEO_DIR}")
    print(f"Output clips will be saved to: {OUTPUT_CLIP_DIR}")
    print(f"Creating disjoint clips of {CLIP_DURATION_S} seconds each.")
    if EXCLUDE_VIDEO_NUMBERS:
        print(f"Excluding videos with numbers: {EXCLUDE_VIDEO_NUMBERS}")
    print("--- Starting processing ---")
    
    start_time = time.time()

    # --- 2. Find all video files ---
    video_files_to_process = []
    for ext in VIDEO_EXTENSIONS:
        # .rglob() searches recursively (in all subfolders)
        video_files_to_process.extend(list(SOURCE_VIDEO_DIR.rglob(f'*{ext}')))
        # Check case-insensitive
        video_files_to_process.extend(list(SOURCE_VIDEO_DIR.rglob(f'*{ext.upper()}')))

    # Remove duplicates if extensions like .mp4 and .MP4 found the same file
    video_files_to_process = sorted(list(set(video_files_to_process)))

    if not video_files_to_process:
        print("Error: No video files found in the source directory.")
        return
        
    print(f"Found {len(video_files_to_process)} video files to process.")

    # --- 3. Process Each Video ---
    total_clips_created = 0
    skipped_videos = 0
    for video_path in video_files_to_process:
        try:
            # Check if this video should be skipped
            if should_skip_video(video_path.name, EXCLUDE_VIDEO_NUMBERS):
                print(f"\n[SKIPPING] Video: {video_path.name} (number in exclusion list)")
                skipped_videos += 1
                continue
                
            print(f"\nProcessing video: {video_path.name}")
            
            # Get the video filename without extension (e.g., "NormalVideo_001")
            video_stem = video_path.stem

            # --- 4. Get duration and calculate number of clips ---
            with VideoFileClip(str(video_path)) as video:
                video_duration = video.duration
            
            if video_duration < MIN_CLIP_DURATION_S:
                print(f"  [Skipping] Video duration ({video_duration:.2f}s) is shorter than minimum clip duration ({MIN_CLIP_DURATION_S}s).")
                continue
                
            # --- 5. Loop and create each clip ---
            clip_start_s = 0.0
            clip_counter = 0
            
            # Loop as long as the remaining video part is longer than our minimum
            while (video_duration - clip_start_s) >= MIN_CLIP_DURATION_S:
                # Calculate end time, clamping it to the video's total duration
                clip_end_s = min(clip_start_s + CLIP_DURATION_S, video_duration)
                
                # e.g., "NormalVideo_001_clip_0.mp4", "NormalVideo_001_clip_1.mp4", ...
                out_name = f"{video_stem}_clip_{clip_counter}.mp4"
                out_path = OUTPUT_CLIP_DIR / out_name
                
                # Call the trim function
                trim_video_disjoint(video_path, out_path, clip_start_s, clip_end_s)
                total_clips_created += 1
                
                # Move to the next clip's start time
                clip_start_s += CLIP_DURATION_S
                clip_counter += 1

            print(f"  Video duration: {video_duration:.2f}s. Created {clip_counter} clip(s).")

        except Exception as e:
            print(f"[ERROR] Unhandled error on video {video_path.name}: {e}")
            # This can happen if a video file is corrupt
            continue
            
    end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"Total videos found: {len(video_files_to_process)}")
    print(f"Videos skipped (in exclusion list): {skipped_videos}")
    print(f"Videos processed: {len(video_files_to_process) - skipped_videos}")
    print(f"Created a total of {total_clips_created} clips.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # --- Dependencies ---
    # This script requires 'moviepy'.
    # You can install it by running:
    # pip install moviepy
    #
    # You also need 'ffmpeg' installed on your system.
    # --------------------
    
    process_disjoint_clips()


