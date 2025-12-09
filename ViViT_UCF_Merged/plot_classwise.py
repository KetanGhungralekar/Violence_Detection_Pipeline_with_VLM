import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- CONFIGURATION ---
# Path to the directory containing your class folders
TARGET_DIR = Path("../UCF_MERGED/Normal") 
TYPE = "Normal"  # Just for title purposes

# List of folder names to strictly IGNORE.
# If empty [], it plots everything.
# Example: IGNORE_CLASSES = ['Normal_Videos_for_Event_Recognition']
IGNORE_CLASSES = [
    'Normal_Videos_for_Event_Recognition', 
]

# File extensions to count
EXTENSIONS = ['*.mp4', '*.avi']
# ---------------------

def plot_class_distribution():
    if not TARGET_DIR.exists():
        print(f"Error: Directory {TARGET_DIR} not found.")
        return

    class_counts = {}
    
    print(f"--- Scanning {TARGET_DIR} ---")
    
    # Iterate over items in the directory
    for item in TARGET_DIR.iterdir():
        if item.is_dir():
            # 1. Check Ignore List
            if item.name in IGNORE_CLASSES:
                continue
            
            # 2. Count files (Non-recursive, usually class folders contain files directly)
            # If your class folders have subfolders, change .glob to .rglob
            count = 0
            for ext in EXTENSIONS:
                count += len(list(item.glob(ext)))
            
            if count > 0:
                class_counts[item.name] = count

    if not class_counts:
        print("No classes found or all were ignored/empty.")
        return

    # Sort by count (Descending) for a better looking graph
    sorted_classes = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
    
    names = list(sorted_classes.keys())
    values = list(sorted_classes.values())

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    # Create bars
    x_pos = np.arange(len(names))
    bars = plt.bar(x_pos, values, color='skyblue', edgecolor='black', alpha=0.7)

    # Labeling
    plt.title(f'Video Count per Class (Total: {sum(values)})', fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # X-Axis Ticks
    plt.xticks(x_pos, names, rotation=45, ha='right', fontsize=10)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.01),
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=9, rotation=0)

    # Add a horizontal line for average if you want
    avg_count = sum(values) / len(values)
    plt.axhline(y=avg_count, color='r', linestyle='--', alpha=0.5, label=f'Avg: {int(avg_count)}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plot_classwise_{TYPE.lower()}.png')
    plt.show()

if __name__ == "__main__":
    plot_class_distribution()