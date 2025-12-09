import os
import json
import argparse

def get_video_files(directory):
    """Get list of video files in the directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)
    return set(video_files)

def main():
    parser = argparse.ArgumentParser(description="Find videos missing in dir1 compared to dir2.")
    parser.add_argument("dir1", help="First directory (where videos might be missing)")
    parser.add_argument("dir2", help="Second directory (reference)")
    parser.add_argument("output_json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    videos_dir1 = get_video_files(args.dir1)
    videos_dir2 = get_video_files(args.dir2)
    
    missing_videos = list(videos_dir2 - videos_dir1)
    
    result = {"missing_videos": missing_videos}
    
    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Found {len(missing_videos)} missing videos. Saved to {args.output_json}")

if __name__ == "__main__":
    main()