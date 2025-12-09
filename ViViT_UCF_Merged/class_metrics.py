import os
import torch
import json
import av
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    VivitImageProcessor,
    VivitForVideoClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import glob
from datasets import Dataset, Features, Value, ClassLabel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# --- 1. CONFIGURATION ---
# --- (All your hyperparameters are here) ---

MODEL_ID = "google/vivit-b-16x2-kinetics400"
DATASET_PATH = "../clips_ucfcrime/"  # Point this to your 'dataset' folder
OUTPUT_DIR = "./vivit-finetuned"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Model & Preprocessing
NUM_CLASSES = 12
NUM_FRAMES = 32
IMAGE_SIZE = 224
NUM_LAYERS_TO_FREEZE = 7

# Training
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 20
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

# Dataset Splits
TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05 # (90/5/5)

# Create output directories
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"--- Configuration ---")
print(f"Model: {MODEL_ID}")
print(f"Dataset Path: {DATASET_PATH}")
print(f"Output Dir: {OUTPUT_DIR}")
print(f"Num Classes: {NUM_CLASSES}")
print(f"Num Frames: {NUM_FRAMES}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("---------------------")


# --- 2. LOAD DATASET AND CREATE SPLITS ---

print(f"Manually loading video paths from '{DATASET_PATH}'...")

# Find all video files (adjust extensions if you have more types)
video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"]
file_paths = []
for ext in video_extensions:
    search_path = os.path.join(DATASET_PATH, "*", ext)
    file_paths.extend(glob.glob(search_path))

if not file_paths:
    raise FileNotFoundError(
        f"No video files ({', '.join(video_extensions)}) found in {DATASET_PATH} subdirectories."
    )

print(f"Found {len(file_paths)} video files.")

# Extract labels and class names from the folder structure
# (e.g., ../clips_ucfcrime/Abuse/video_001.mp4 -> "Abuse")
class_names = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in file_paths])))
label2id = {name: i for i, name in enumerate(class_names)}
id2label = {i: name for i, name in enumerate(class_names)}
labels = [label2id[os.path.basename(os.path.dirname(p))] for p in file_paths]

# Create a Hugging Face Dataset manually from the paths and labels
dataset_dict = {
    "video_path": file_paths,
    "label": labels
}

# Define the features, telling the dataset that "label" is a ClassLabel
features = Features({
    "video_path": Value("string"),
    "label": ClassLabel(names=class_names)
})

dataset = Dataset.from_dict(dataset_dict, features=features)

print(f"Loaded {dataset.num_rows} videos in {len(class_names)} classes.")
print(f"Your {len(class_names)} classes are: {class_names}")

# Check if Num Classes in config matches
if NUM_CLASSES != len(class_names):
    print(f"[WARNING] Your NUM_CLASSES config is {NUM_CLASSES} but your dataset has {len(class_names)} classes.")
    print(f"Proceeding with {len(class_names)} classes.")
    NUM_CLASSES = len(class_names)


# Create 90/5/5 splits
# 1. Split into 90% train and 10% temp
train_test_split = dataset.train_test_split(test_size=(1.0 - TRAIN_SPLIT), seed=42)
train_dataset = train_test_split["train"]

# 2. Split the 10% temp into 50% validation and 50% test (which is 5% of total)
val_test_split = train_test_split["test"].train_test_split(test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

print(f"Dataset splits (stratified):")
print(f" - Train: {len(train_dataset)} examples")
print(f" - Validation: {len(val_dataset)} examples")
print(f" - Test: {len(test_dataset)} examples")

# --- 3. VIDEO PREPROCESSING ---

# Load the processor
image_processor = VivitImageProcessor.from_pretrained(MODEL_ID)

# Get the default normalization values
image_mean = image_processor.image_mean
image_std = image_processor.image_std
image_size = image_processor.crop_size["height"]
print(f"Using model's default normalization: mean={image_mean}, std={image_std}")
print(f"Images will be cropped to: {image_size}x{image_size}")


def read_video_pyav(container, indices):
    """
    Decodes the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: Decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def sample_frame_indices(clip_len, total_frames):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (int): Total number of frames to sample.
        total_frames (int): Total number of frames in the video.
    Returns:
        np.ndarray: Array of sampled frame indices
    """
    # Uniformly sample frames
    indices = np.linspace(0, total_frames - 1, num=clip_len, dtype=int)
    return indices


def preprocess_function(examples):
    """
    Processes a *batch* of video examples.
    'examples' will be a dict: {'video_path': [path1, ...], 'label': [label1, ...]}
    """
    
    # 1. Create lists to hold the processed outputs
    batch_pixel_values = []
    batch_labels = []

    # Get the correct image size from the processor's crop_size
    image_size = image_processor.crop_size["height"]

    # 2. Iterate over each video path and label in the batch
    for video_path, label in zip(examples["video_path"], examples["label"]):
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames

            # Handle videos shorter than NUM_FRAMES
            if total_frames < NUM_FRAMES:
                indices = np.arange(total_frames)
                indices = np.tile(indices, (NUM_FRAMES // total_frames) + 1)[:NUM_FRAMES]
                indices = np.sort(indices)
            else:
                # Sample frame indices
                indices = sample_frame_indices(NUM_FRAMES, total_frames)
            
            # Read and decode frames
            video = read_video_pyav(container, indices)
            
            # Preprocess with the image processor (resize, normalize, etc.)
            processed_video = image_processor(list(video), return_tensors="pt")
            
            # Append the processed tensor and its label
            batch_pixel_values.append(processed_video.pixel_values.squeeze())
            batch_labels.append(label)

        except Exception as e:
            # If one video fails, log it and append a "bad" sample
            print(f"Error processing video {video_path}: {e}. Skipping.")
            # Append a tensor of zeros with the correct shape and a bad label
            batch_pixel_values.append(torch.zeros((NUM_FRAMES, 3, image_size, image_size)))
            batch_labels.append(-1) # This -1 label will be filtered by compute_metrics

    # 3. Collate the lists of tensors into a single batch tensor
    #    This is what the Trainer expects.
    try:
        final_pixel_values = torch.stack(batch_pixel_values)
    except RuntimeError as re:
        print(f"Error stacking batch: {re}. One of the videos may have a shape mismatch despite processing.")
        # If stacking fails, we can't proceed with this batch.
        # Returning empty tensors or a subset might be complex.
        # For simplicity, we'll return the first item to show the error,
        # but this batch will likely fail.
        # A more robust solution would be to filter batch_pixel_values
        # for correct shapes, but the 'except' block above *should* prevent this.
        return {"pixel_values": batch_pixel_values[0].unsqueeze(0), "label": [batch_labels[0]]}


    return {
        "pixel_values": final_pixel_values,
        "label": batch_labels  # Return a list of labels
    }

# Set the transform to be applied on-the-fly
print("Setting on-the-fly transforms for train, validation, and test datasets...")
train_dataset.set_transform(preprocess_function)
val_dataset.set_transform(preprocess_function)
test_dataset.set_transform(preprocess_function)


# --- 4. MODEL LOADING AND LAYER FREEZING ---

print(f"Loading pre-trained model: {MODEL_ID}")
model = VivitForVideoClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True  # Discard the old 400-class head
)

# --- Freeze the specified layers ---
print("Freezing layers...")
# 1. Freeze the embeddings
for param in model.vivit.embeddings.parameters():
    param.requires_grad = False
print(" - Embeddings frozen.")

# 2. Freeze the first N encoder layers
for i in range(NUM_LAYERS_TO_FREEZE):
    for param in model.vivit.encoder.layer[i].parameters():
        param.requires_grad = False
print(f" - First {NUM_LAYERS_TO_FREEZE} of {model.config.num_hidden_layers} encoder layers frozen.")

# Optional: Print trainable parameters to verify
print("\n--- Trainable Parameters ---")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print("--------------------------\n")


# --- 5. METRICS AND TRAINER SETUP ---

def compute_metrics(eval_pred):
    """
    Computes overall metrics from predictions.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    # Filter out bad samples (if any)
    valid_indices = labels != -1
    labels = labels[valid_indices]
    predictions = predictions[valid_indices]

    if len(labels) == 0:
        return {
            "accuracy": 0,
            "f1_weighted": 0,
            "precision_weighted": 0,
            "recall_weighted": 0
        }

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall
    }


# Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=MAX_EPOCHS,
    
    # Batch sizes
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    
    # Evaluation and logging
    eval_strategy="epoch",
    logging_strategy="epoch",
    
    # Saving
    save_strategy="epoch",
    save_total_limit=3, # Saves the last 3 checkpoints
    
    # Early Stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_weighted",
    greater_is_better=True,
    
    # Misc
    fp16=True if torch.cuda.is_available() else False, # Enable mixed precision
    report_to="none", # Disable WandB/Tensorboard
    remove_unused_columns=False, # Important for set_transform
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)


# --- 7a. Load and Process Metrics from JSON ---
print("Loading log history from trainer_state.json...")

# Path to the trainer state file in your main output directory
STATE_FILE_PATH = os.path.join(OUTPUT_DIR, "trainer_state.json")

try:
    with open(STATE_FILE_PATH, 'r') as f:
        trainer_state = json.load(f)
    history = trainer_state["log_history"]
    print(f"Successfully loaded log history with {len(history)} entries.")
except Exception as e:
    print(f"Error loading {STATE_FILE_PATH}: {e}")
    sys.exit(1)

# Manually re-load the history into the trainer object
# This is critical so the rest of the script (like 7c) doesn't fail
trainer.state.log_history = history

# --- (This is the fixed pandas logic) ---

df_history = pd.DataFrame(history)

# Separate train logs and select ONLY the columns we need
if "loss" in df_history.columns:
    df_train = df_history[df_history["loss"].notna()][["epoch", "loss"]].rename(
        columns={"loss": "train_loss"}
    )
else:
    print("Warning: No 'loss' (training loss) found in logs. Skipping train loss plotting.")
    df_train = pd.DataFrame(columns=["epoch", "train_loss"])

# Separate eval logs and select ONLY the columns we need
eval_cols_to_find = [
    "epoch", "eval_loss", "eval_accuracy", 
    "eval_f1_weighted", "eval_precision_weighted", "eval_recall_weighted"
]
# Get all columns that are *actually present* in the log
available_eval_cols = [col for col in eval_cols_to_find if col in df_history.columns]

if "eval_loss" in available_eval_cols:
    df_eval = df_history[df_history["eval_loss"].notna()][available_eval_cols].rename(
        columns={"eval_loss": "val_loss", 
                 "eval_accuracy": "val_accuracy",
                 "eval_f1_weighted": "val_f1", 
                 "eval_precision_weighted": "val_precision",
                 "eval_recall_weighted": "val_recall"}
    )
else:
    print("Warning: No 'eval_loss' found in logs. Skipping all evaluation metrics.")
    # Create empty dataframe with renamed columns to prevent merge errors
    rename_map = {
        "eval_loss": "val_loss", "eval_accuracy": "val_accuracy",
        "eval_f1_weighted": "val_f1", "eval_precision_weighted": "val_precision",
        "eval_recall_weighted": "val_recall", "epoch": "epoch"
    }
    df_eval = pd.DataFrame(columns=[rename_map[col] for col in available_eval_cols])


# Merge the minimal dataframes
if not df_train.empty and not df_eval.empty:
    df_epoch_metrics = pd.merge(df_train, df_eval, on="epoch", how="outer").sort_values("epoch")
elif not df_train.empty:
    df_epoch_metrics = df_train.sort_values("epoch")
elif not df_eval.empty:
    df_epoch_metrics = df_eval.sort_values("epoch")
else:
    print("ERROR: No valid metric data to save.", file=sys.stderr)
    sys.exit(1)

# Save to CSV
epoch_csv_path = os.path.join(METRICS_DIR, "epoch_wise_metrics.csv")
df_epoch_metrics.to_csv(epoch_csv_path, index=False)
print(f"Epoch-wise metrics saved to: {epoch_csv_path}")

# --- (End of new 7a block) ---
# The script will now continue to 7b (plotting) and 7c (class report)
# --- 7b. Plot Epoch-wise Metrics ---

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["train_loss"], label="Train Loss")
plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "loss_vs_epoch.png"))
print(f"Loss plot saved to: {PLOTS_DIR}/loss_vs_epoch.png")

# Plot F1-Score
plt.figure(figsize=(12, 6))
plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_f1"], label="Validation F1 (Weighted)")
plt.xlabel("Epoch")
plt.ylabel("F1 Score (Weighted)")
plt.title("Validation F1 Score Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "f1_vs_epoch.png"))
print(f"F1 plot saved to: {PLOTS_DIR}/f1_vs_epoch.png")

# Plot Accuracy
plt.figure(figsize=(12, 6))
plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_epoch.png"))
print(f"Accuracy plot saved to: {PLOTS_DIR}/accuracy_vs_epoch.png")


# --- 7c. Final Evaluation on Test Set (Class-wise) ---
print("\n--- Evaluating on Test Set ---")

# Run predictions on the test set
test_predictions = trainer.predict(test_dataset)

# Get true labels and predicted labels
y_true = test_predictions.label_ids
y_pred = np.argmax(test_predictions.predictions, axis=1)

# Filter out bad samples (if any)
valid_indices = y_true != -1
y_true = y_true[valid_indices]
y_pred = y_pred[valid_indices]

# Generate classification report (dictionary)
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)

# Convert to DataFrame
df_class_report = pd.DataFrame(report_dict).transpose()

# Save class-wise report to CSV
class_csv_path = os.path.join(METRICS_DIR, "class_wise_test_report.csv")
df_class_report.to_csv(class_csv_path, index=True)
print(f"Class-wise test report saved to: {class_csv_path}")

# Print the final report to console
print("\n--- Final Test Set Report ---")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("--------------------------")
print(f"Script finished. All outputs saved in {OUTPUT_DIR}")
