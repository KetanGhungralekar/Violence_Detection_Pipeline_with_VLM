import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns  # Optional, for better styling

# --- CONFIGURATION ---
# Path to the parent directory containing the two main folders
DATASET_DIR = Path("../UCF_MERGED") 

# Exact folder names for the two categories
# Update these to match your actual directory names
CLASS_MAPPING = {
    'Normal': 'Normal',   # key is label, value is folder name
    'Abnormal': 'Abnormal'      # key is label, value is folder name
}

# File extensions to count
EXTENSIONS = ['*.mp4', '*.avi', '*.mkv']
# ---------------------

def count_files_recursively(directory):
    """Counts all video files in a directory and its subdirectories."""
    count = 0
    if not directory.exists():
        print(f"Warning: Directory not found: {directory}")
        return 0
        
    for ext in EXTENSIONS:
        # rglob searches recursively through all subfolders
        count += len(list(directory.rglob(ext)))
    return count

def plot_binary_distribution():
    counts = {}
    
    print("--- Counting Files ---")
    for label, folder_name in CLASS_MAPPING.items():
        full_path = DATASET_DIR / folder_name
        num_files = count_files_recursively(full_path)
        counts[label] = num_files
        print(f"{label} ({folder_name}): {num_files}")

    # Plotting
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 6))
    # Use a specific color palette
    bars = plt.bar(labels, values, color=['#4CAF50', '#F44336'], alpha=0.8)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'Dataset Distribution: Normal vs Abnormal', fontsize=14)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plot_normal_abnormal.png')
    plt.show()

if __name__ == "__main__":
    # Requires matplotlib
    # pip install matplotlib
    plot_binary_distribution()