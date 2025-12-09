import os
import cv2
from tqdm import tqdm
import json

# ===========================
# CONFIGURATION
# ===========================
DATASET_ROOT = "../UCF_MERGED_NORM/"  # Root folder containing Normal and Abnormal
EXPECTED_FRAMES = 32

# ===========================
# FRAME CHECKER
# ===========================
def count_video_frames(video_path):
    """Count the number of frames in a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1, "Could not open video"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count, None
    except Exception as e:
        return -1, str(e)

# ===========================
# MAIN PIPELINE
# ===========================
results = {
    "correct": [],
    "incorrect": [],
    "errors": []
}

total_videos = 0
correct_count = 0
incorrect_count = 0
error_count = 0

print(f"🔍 Checking videos in {DATASET_ROOT}")
print(f"Expected frames per video: {EXPECTED_FRAMES}\n")

for label in ["Normal", "Abnormal"]:
    label_path = os.path.join(DATASET_ROOT, label)
    if not os.path.isdir(label_path):
        print(f"⚠️  Warning: {label} folder not found, skipping...")
        continue

    for action_class in os.listdir(label_path):
        action_path = os.path.join(label_path, action_class)
        if not os.path.isdir(action_path):
            continue

        print(f"📂 Checking {label}/{action_class}")
        
        video_files = [f for f in os.listdir(action_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for file in tqdm(video_files, desc=f"{label}/{action_class}"):
            video_path = os.path.join(action_path, file)
            total_videos += 1
            
            frame_count, error = count_video_frames(video_path)
            
            if error:
                results["errors"].append({
                    "path": f"{label}/{action_class}/{file}",
                    "error": error
                })
                error_count += 1
            elif frame_count == EXPECTED_FRAMES:
                results["correct"].append(f"{label}/{action_class}/{file}")
                correct_count += 1
            else:
                results["incorrect"].append({
                    "path": f"{label}/{action_class}/{file}",
                    "frames": frame_count,
                    "expected": EXPECTED_FRAMES
                })
                incorrect_count += 1

# ===========================
# REPORT
# ===========================
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)
print(f"Total videos checked: {total_videos}")
print(f"✅ Correct ({EXPECTED_FRAMES} frames): {correct_count} ({correct_count/total_videos*100:.1f}%)")
print(f"❌ Incorrect frame count: {incorrect_count} ({incorrect_count/total_videos*100:.1f}%)")
print(f"⚠️  Errors: {error_count} ({error_count/total_videos*100:.1f}%)")

# Save detailed results to JSON
output_file = "frame_check_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n📄 Detailed results saved to: {output_file}")

# Print some examples of issues
if incorrect_count > 0:
    print(f"\n❌ Examples of videos with incorrect frame count (showing first 10):")
    for item in results["incorrect"][:10]:
        print(f"   {item['path']}: {item['frames']} frames (expected {item['expected']})")

if error_count > 0:
    print(f"\n⚠️  Examples of videos with errors (showing first 10):")
    for item in results["errors"][:10]:
        print(f"   {item['path']}: {item['error']}")
