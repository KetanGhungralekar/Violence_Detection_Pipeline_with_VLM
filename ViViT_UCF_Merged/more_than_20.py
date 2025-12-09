import os
from pathlib import Path
from moviepy.editor import VideoFileClip

# --- User Configuration ---
# Path to the directory to search (will search all subdirectories)
SEARCH_DIR = Path("../clips_ucfcrime/")

# Duration threshold in seconds
DURATION_THRESHOLD = 20.0

# Video file extensions to check
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
# --------------------------

def find_long_clips(search_dir: Path, threshold: float):
    """
    Searches through all subdirectories and finds clips longer than threshold.
    """
    if not search_dir.exists() or not search_dir.is_dir():
        print(f"Error: Directory not found at {search_dir}")
        return
    
    print(f"Searching for clips longer than {threshold} seconds in: {search_dir}")
    print("=" * 60)
    
    long_clips = []
    total_videos_checked = 0
    
    # Find all video files recursively
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(search_dir.rglob(f'*{ext}')))
        video_files.extend(list(search_dir.rglob(f'*{ext.upper()}')))
    
    # Remove duplicates
    video_files = sorted(list(set(video_files)))
    
    print(f"Found {len(video_files)} video files to check.\n")
    
    # Check each video
    for video_path in video_files:
        try:
            total_videos_checked += 1
            
            # Get video duration
            with VideoFileClip(str(video_path)) as video:
                duration = video.duration
            
            # If longer than threshold, add to list
            if duration > threshold:
                # Get relative path from search directory
                relative_path = video_path.relative_to(search_dir)
                long_clips.append({
                    'name': video_path.name,
                    'path': str(relative_path),
                    'subdir': video_path.parent.name,
                    'duration': duration
                })
                print(f"✓ {relative_path} - Duration: {duration:.2f}s")
                
        except Exception as e:
            print(f"✗ Error checking {video_path.name}: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total videos checked: {total_videos_checked}")
    print(f"Clips longer than {threshold} seconds: {len(long_clips)}")
    print("\nList of long clips:")
    print("-" * 60)
    
    if long_clips:
        # Group by subdirectory
        from collections import defaultdict
        clips_by_subdir = defaultdict(list)
        
        for clip in long_clips:
            clips_by_subdir[clip['subdir']].append(clip)
        
        for subdir, clips in sorted(clips_by_subdir.items()):
            print(f"\n[{subdir}] - {len(clips)} clip(s)")
            for clip in clips:
                print(f"  - {clip['name']} ({clip['duration']:.2f}s)")
    else:
        print("No clips found longer than the threshold.")
    
    # Return the list
    return long_clips

if __name__ == "__main__":
    # Dependencies: pip install moviepy
    
    long_clips_list = find_long_clips(SEARCH_DIR, DURATION_THRESHOLD)
    
    # Optional: Save to file
    if long_clips_list:
        output_file = SEARCH_DIR / "long_clips_report.txt"
        with open(output_file, 'w') as f:
            f.write(f"Clips longer than {DURATION_THRESHOLD} seconds:\n\n")
            for clip in long_clips_list:
                f.write(f"{clip['path']} - {clip['duration']:.2f}s\n")
        print(f"\nReport saved to: {output_file}")