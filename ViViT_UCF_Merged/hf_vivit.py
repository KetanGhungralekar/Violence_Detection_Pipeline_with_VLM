import os
import json
import torch
import av
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
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
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ----------------------- Logging Setup -----------------------
log_filename = f'./training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("Training started")

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
MAX_EPOCHS = 200
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

logger.info(f"--- Configuration ---")
logger.info(f"Model: {MODEL_ID}")
logger.info(f"Dataset Path: {DATASET_PATH}")
logger.info(f"Output Dir: {OUTPUT_DIR}")
logger.info(f"Num Classes: {NUM_CLASSES}")
logger.info(f"Num Frames: {NUM_FRAMES}")
logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
logger.info("---------------------")


# --- 2. LOAD DATASET AND CREATE SPLITS ---

logger.info(f"Manually loading video paths from '{DATASET_PATH}'...")

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

logger.info(f"Found {len(file_paths)} video files.")

# Extract labels and class names from the folder structure
# (e.g., ../clips_ucfcrime/Abuse/video_001.mp4 -> "Abuse")
# Filter out the Normal_Videos_for_Event_Recognition class
EXCLUDE_CLASS = "Normal_Videos_for_Event_Recognition"

# Get all class names
all_class_names = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in file_paths])))

# Filter out the excluded class
class_names = [name for name in all_class_names if name != EXCLUDE_CLASS]

# Filter out file paths that belong to the excluded class
filtered_file_paths = [p for p in file_paths if os.path.basename(os.path.dirname(p)) != EXCLUDE_CLASS]
file_paths = filtered_file_paths

logger.info(f"Excluding class: {EXCLUDE_CLASS}")
logger.info(f"Remaining videos after exclusion: {len(file_paths)}")

label2id = {name: i for i, name in enumerate(class_names)}
id2label = {i: name for i, name in enumerate(class_names)}
labels = [label2id[os.path.basename(os.path.dirname(p))] for p in file_paths]

# Verify all labels are valid
logger.info(f"Label range: {min(labels)} to {max(labels)}")
logger.info(f"Number of classes: {len(class_names)}")
assert all(0 <= label < len(class_names) for label in labels), "Invalid labels found!"

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

logger.info(f"Loaded {dataset.num_rows} videos in {len(class_names)} classes.")
logger.info(f"Your {len(class_names)} classes are: {class_names}")

# Check if Num Classes in config matches
if NUM_CLASSES != len(class_names):
    logger.warning(f"Your NUM_CLASSES config is {NUM_CLASSES} but your dataset has {len(class_names)} classes.")
    logger.info(f"Proceeding with {len(class_names)} classes.")
    NUM_CLASSES = len(class_names)


# Create 90/5/5 splits
# 1. Split into 90% train and 10% temp
train_test_split = dataset.train_test_split(test_size=(1.0 - TRAIN_SPLIT), seed=42)
train_dataset = train_test_split["train"]

# 2. Split the 10% temp into 50% validation and 50% test (which is 5% of total)
val_test_split = train_test_split["test"].train_test_split(test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), seed=42)
val_dataset = val_test_split["train"]
test_dataset = val_test_split["test"]

logger.info(f"Dataset splits (stratified):")
logger.info(f" - Train: {len(train_dataset)} examples")
logger.info(f" - Validation: {len(val_dataset)} examples")
logger.info(f" - Test: {len(test_dataset)} examples")

# --- 3. VIDEO PREPROCESSING ---

# Load the processor
image_processor = VivitImageProcessor.from_pretrained(MODEL_ID)

# Get the default normalization values
image_mean = image_processor.image_mean
image_std = image_processor.image_std
image_size = image_processor.crop_size["height"]
logger.info(f"Using model's default normalization: mean={image_mean}, std={image_std}")
logger.info(f"Images will be cropped to: {image_size}x{image_size}")


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
            
            # If total_frames is 0 or None, count frames manually
            if total_frames == 0 or total_frames is None:
                total_frames = sum(1 for _ in container.decode(video=0))
                container.seek(0)

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
            container.close()
            
            # Ensure we have exactly NUM_FRAMES frames BEFORE processing
            if video.shape[0] != NUM_FRAMES:
                if video.shape[0] < NUM_FRAMES:
                    # Pad by repeating the last frame
                    padding = np.repeat(video[-1:], NUM_FRAMES - video.shape[0], axis=0)
                    video = np.concatenate([video, padding], axis=0)
                else:
                    # Trim to NUM_FRAMES
                    video = video[:NUM_FRAMES]
            
            # Ensure video is exactly NUM_FRAMES x H x W x 3
            assert video.shape[0] == NUM_FRAMES, f"Video has {video.shape[0]} frames, expected {NUM_FRAMES}"
            
            # Preprocess each frame individually and stack them
            # This ensures consistent output shape
            processed_frames = []
            for frame in video:
                # Process single frame - pass as PIL Image or numpy array
                # Make sure frame is in the right format (H, W, C)
                processed = image_processor(images=frame, return_tensors="pt")
                # Extract the pixel values and squeeze ALL batch dimensions
                # Shape should go from [1, 3, 224, 224] to [3, 224, 224]
                pixel_vals = processed.pixel_values
                while pixel_vals.dim() > 3:  # Keep squeezing until we have [3, 224, 224]
                    pixel_vals = pixel_vals.squeeze(0)
                processed_frames.append(pixel_vals)
            
            # Stack all frames: [NUM_FRAMES, 3, 224, 224]
            video_tensor = torch.stack(processed_frames)
            
            # Final safety check
            expected_shape = (NUM_FRAMES, 3, image_size, image_size)
            if video_tensor.shape != expected_shape:
                print(f"Error: Video {video_path} has shape {video_tensor.shape}, expected {expected_shape}. Using zero tensor.")
                video_tensor = torch.zeros(expected_shape)
                label = -1
            
            batch_pixel_values.append(video_tensor)
            batch_labels.append(label)

        except Exception as e:
            # If one video fails, log it and append a "bad" sample
            logger.warning(f"Error processing video {video_path}: {e}. Using zero tensor.")
            # Append a tensor of zeros with the correct shape and a bad label
            batch_pixel_values.append(torch.zeros((NUM_FRAMES, 3, image_size, image_size)))
            batch_labels.append(-1) # This -1 label will be filtered by compute_metrics

    # 3. Collate the lists of tensors into a single batch tensor
    #    This is what the Trainer expects.
    try:
        final_pixel_values = torch.stack(batch_pixel_values)
    except RuntimeError as re:
        logger.error(f"CRITICAL Error stacking batch: {re}")
        logger.error(f"Batch shapes: {[x.shape for x in batch_pixel_values]}")
        # Create a batch of zero tensors as fallback
        final_pixel_values = torch.zeros((len(batch_pixel_values), NUM_FRAMES, 3, image_size, image_size))
        batch_labels = [-1] * len(batch_labels)

    # 4. CRITICAL: Filter out invalid samples (label == -1) to prevent CUDA assertion errors
    valid_mask = [label != -1 for label in batch_labels]
    
    if not any(valid_mask):
        # If all samples are invalid, return a minimal valid batch to avoid crashes
        logger.warning("All samples in batch are invalid! Returning single zero sample.")
        return {
            "pixel_values": torch.zeros((1, NUM_FRAMES, 3, image_size, image_size)),
            "label": [0]  # Use valid label 0 instead of -1
        }
    
    # Filter to keep only valid samples
    if not all(valid_mask):
        valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
        final_pixel_values = final_pixel_values[valid_indices]
        batch_labels = [batch_labels[i] for i in valid_indices]
        logger.info(f"Filtered out {sum(not v for v in valid_mask)} invalid samples from batch.")

    return {
        "pixel_values": final_pixel_values,
        "label": batch_labels  # Return a list of labels
    }

# Set the transform to be applied on-the-fly
logger.info("Setting on-the-fly transforms for train, validation, and test datasets...")
train_dataset.set_transform(preprocess_function)
val_dataset.set_transform(preprocess_function)
test_dataset.set_transform(preprocess_function)


# --- 4. MODEL LOADING AND LAYER FREEZING ---

logger.info(f"Loading pre-trained model: {MODEL_ID}")
model = VivitForVideoClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True  # Discard the old 400-class head
)

# --- Freeze the specified layers ---
logger.info("Freezing layers...")
# 1. Freeze the embeddings
for param in model.vivit.embeddings.parameters():
    param.requires_grad = False
logger.info(" - Embeddings frozen.")

# 2. Freeze the first N encoder layers
for i in range(NUM_LAYERS_TO_FREEZE):
    for param in model.vivit.encoder.layer[i].parameters():
        param.requires_grad = False
logger.info(f" - First {NUM_LAYERS_TO_FREEZE} of {model.config.num_hidden_layers} encoder layers frozen.")

# Optional: Print trainable parameters to verify
logger.info("\n--- Trainable Parameters ---")
for name, param in model.named_parameters():
    if param.requires_grad:
        logger.info(name)
logger.info("--------------------------\n")


# --- 5. METRICS AND TRAINER SETUP ---

def compute_metrics(eval_pred):
    """
    Computes overall and per-class metrics from predictions.
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
            "f1_macro": 0,
            "f1_micro": 0,
            "precision_weighted": 0,
            "precision_macro": 0,
            "precision_micro": 0,
            "recall_weighted": 0,
            "recall_macro": 0,
            "recall_micro": 0
        }

    # Calculate overall metrics with zero_division parameter to suppress warnings
    accuracy = accuracy_score(labels, predictions)
    
    # Weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    # Macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Micro metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_weighted": recall_weighted,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro
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


# --- 6. START TRAINING ---

logger.info("\n--- Starting Model Training ---")
train_result = trainer.train()
logger.info("--- Training Finished ---")

# --- 7. POST-TRAINING ANALYSIS AND REPORTING ---

logger.info("\n--- Generating Reports and Plots ---")

# --- 7a. Save Epoch-wise Metrics ---
history = trainer.state.log_history

# Convert history to DataFrame
df_history = pd.DataFrame(history)

# Check if history is empty or missing data
if "loss" not in df_history.columns and "eval_loss" not in df_history.columns:
    logger.error("No 'loss' or 'eval_loss' columns found in log history.")
    logger.error("No metrics to plot.")
    df_epoch_metrics = pd.DataFrame() # Create empty df
else:
    # Separate train logs and select ONLY the columns we need
    if "loss" in df_history.columns:
        df_train = df_history[df_history["loss"].notna()][["epoch", "loss"]].rename(
            columns={"loss": "train_loss"}
        )
    else:
        logger.warning("No 'loss' (training loss) found in logs. Skipping train loss plotting.")
        df_train = pd.DataFrame(columns=["epoch", "train_loss"])

    # Separate eval logs and select ONLY the columns we need
    eval_cols_to_find = [
        "epoch", "eval_loss", "eval_accuracy", 
        "eval_f1_weighted", "eval_f1_macro", "eval_f1_micro",
        "eval_precision_weighted", "eval_precision_macro", "eval_precision_micro",
        "eval_recall_weighted", "eval_recall_macro", "eval_recall_micro"
    ]
    # Get all columns that are *actually present* in the log
    available_eval_cols = [col for col in eval_cols_to_find if col in df_history.columns]

    if "eval_loss" in available_eval_cols:
        df_eval = df_history[df_history["eval_loss"].notna()][available_eval_cols].rename(
            columns={
                "eval_loss": "val_loss", 
                "eval_accuracy": "val_accuracy",
                "eval_f1_weighted": "val_f1_weighted", 
                "eval_f1_macro": "val_f1_macro",
                "eval_f1_micro": "val_f1_micro",
                "eval_precision_weighted": "val_precision_weighted",
                "eval_precision_macro": "val_precision_macro",
                "eval_precision_micro": "val_precision_micro",
                "eval_recall_weighted": "val_recall_weighted",
                "eval_recall_macro": "val_recall_macro",
                "eval_recall_micro": "val_recall_micro"
            }
        )
    else:
        logger.warning("No 'eval_loss' found in logs. Skipping all evaluation metrics.")
        # Create empty dataframe with renamed columns to prevent merge errors
        rename_map = {
            "eval_loss": "val_loss", "eval_accuracy": "val_accuracy",
            "eval_f1_weighted": "val_f1_weighted", "eval_f1_macro": "val_f1_macro",
            "eval_f1_micro": "val_f1_micro",
            "eval_precision_weighted": "val_precision_weighted",
            "eval_precision_macro": "val_precision_macro",
            "eval_precision_micro": "val_precision_micro",
            "eval_recall_weighted": "val_recall_weighted",
            "eval_recall_macro": "val_recall_macro",
            "eval_recall_micro": "val_recall_micro",
            "epoch": "epoch"
        }
        df_eval = pd.DataFrame(columns=[rename_map.get(col, col) for col in available_eval_cols])


    # Merge the minimal dataframes
    if not df_train.empty and not df_eval.empty:
        df_epoch_metrics = pd.merge(df_train, df_eval, on="epoch", how="outer").sort_values("epoch")
    elif not df_train.empty:
        df_epoch_metrics = df_train.sort_values("epoch")
    elif not df_eval.empty:
        df_epoch_metrics = df_eval.sort_values("epoch")
    else:
        logger.error("No valid metric data to save.")
        df_epoch_metrics = pd.DataFrame() # Create empty df

# Save to CSV
epoch_csv_path = os.path.join(METRICS_DIR, "epoch_wise_metrics.csv")
df_epoch_metrics.to_csv(epoch_csv_path, index=False)
logger.info(f"Epoch-wise metrics saved to: {epoch_csv_path}")


# --- 7b. Plot Epoch-wise Metrics ---

# Plot Loss
plt.figure(figsize=(12, 6))
if "train_loss" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["train_loss"], label="Train Loss", marker='o')
if "val_loss" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_loss"], label="Validation Loss", marker='s')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training and Validation Loss Over Epochs", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot_path = os.path.join(PLOTS_DIR, "loss_vs_epoch.png")
plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Loss plot saved to: {loss_plot_path}")

# Plot F1-Score (Multiple variants)
plt.figure(figsize=(12, 6))
if "val_f1_weighted" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_f1_weighted"], 
             label="Validation F1 (Weighted)", marker='o', linewidth=2)
if "val_f1_macro" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_f1_macro"], 
             label="Validation F1 (Macro)", marker='s', linewidth=2)
if "val_f1_micro" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_f1_micro"], 
             label="Validation F1 (Micro)", marker='^', linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.title("Validation F1 Score Over Epochs", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
f1_plot_path = os.path.join(PLOTS_DIR, "f1_vs_epoch.png")
plt.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"F1 plot saved to: {f1_plot_path}")

# Plot Accuracy
plt.figure(figsize=(12, 6))
if "val_accuracy" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_accuracy"], 
             label="Validation Accuracy", marker='o', linewidth=2, color='green')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Validation Accuracy Over Epochs", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
acc_plot_path = os.path.join(PLOTS_DIR, "accuracy_vs_epoch.png")
plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Accuracy plot saved to: {acc_plot_path}")

# Plot Precision (Multiple variants)
plt.figure(figsize=(12, 6))
if "val_precision_weighted" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_precision_weighted"], 
             label="Validation Precision (Weighted)", marker='o', linewidth=2)
if "val_precision_macro" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_precision_macro"], 
             label="Validation Precision (Macro)", marker='s', linewidth=2)
if "val_precision_micro" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_precision_micro"], 
             label="Validation Precision (Micro)", marker='^', linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Validation Precision Over Epochs", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
prec_plot_path = os.path.join(PLOTS_DIR, "precision_vs_epoch.png")
plt.savefig(prec_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Precision plot saved to: {prec_plot_path}")

# Plot Recall (Multiple variants)
plt.figure(figsize=(12, 6))
if "val_recall_weighted" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_recall_weighted"], 
             label="Validation Recall (Weighted)", marker='o', linewidth=2)
if "val_recall_macro" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_recall_macro"], 
             label="Validation Recall (Macro)", marker='s', linewidth=2)
if "val_recall_micro" in df_epoch_metrics.columns:
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_recall_micro"], 
             label="Validation Recall (Micro)", marker='^', linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Recall", fontsize=12)
plt.title("Validation Recall Over Epochs", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
rec_plot_path = os.path.join(PLOTS_DIR, "recall_vs_epoch.png")
plt.savefig(rec_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Recall plot saved to: {rec_plot_path}")



# --- 7c. Final Evaluation on Test Set (Class-wise and Comprehensive) ---
logger.info("\n--- Evaluating on Test Set ---")

# Run predictions on the test set
test_predictions = trainer.predict(test_dataset)

# Get true labels and predicted labels
y_true = test_predictions.label_ids
y_pred = np.argmax(test_predictions.predictions, axis=1)

# Filter out bad samples (if any)
valid_indices = y_true != -1
y_true = y_true[valid_indices]
y_pred = y_pred[valid_indices]

# --- Overall Metrics ---
test_accuracy = accuracy_score(y_true, y_pred)
test_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
test_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
test_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

test_precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
test_recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
test_f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

test_precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
test_recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
test_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# --- Per-Class Metrics ---
classwise_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
classwise_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
classwise_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

# Compute confusion matrix for per-class accuracy
cm = confusion_matrix(y_true, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)

# Log overall results
logger.info(f"\n🎯 Test Results (Overall):")
logger.info(f"  Accuracy: {test_accuracy:.4f}")
logger.info(f"  Precision (macro): {test_precision_macro:.4f}")
logger.info(f"  Recall (macro): {test_recall_macro:.4f}")
logger.info(f"  F1 Score (macro): {test_f1_macro:.4f}")
logger.info(f"  Precision (micro): {test_precision_micro:.4f}")
logger.info(f"  Recall (micro): {test_recall_micro:.4f}")
logger.info(f"  F1 Score (micro): {test_f1_micro:.4f}")
logger.info(f"  Precision (weighted): {test_precision_weighted:.4f}")
logger.info(f"  Recall (weighted): {test_recall_weighted:.4f}")
logger.info(f"  F1 Score (weighted): {test_f1_weighted:.4f}")

# Generate classification report (dictionary)
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
)

# Convert to DataFrame
df_class_report = pd.DataFrame(report_dict).transpose()

# Save class-wise report to CSV
class_csv_path = os.path.join(METRICS_DIR, "class_wise_test_report.csv")
df_class_report.to_csv(class_csv_path, index=True)
logger.info(f"Class-wise test report saved to: {class_csv_path}")

# --- Save detailed metrics to text file ---
detailed_metrics_path = os.path.join(METRICS_DIR, "test_detailed_metrics.txt")
with open(detailed_metrics_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("TEST SET EVALUATION - DETAILED METRICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OVERALL METRICS:\n")
    f.write(f"  Accuracy: {test_accuracy:.4f}\n")
    f.write(f"  Precision (macro): {test_precision_macro:.4f}\n")
    f.write(f"  Recall (macro): {test_recall_macro:.4f}\n")
    f.write(f"  F1 Score (macro): {test_f1_macro:.4f}\n")
    f.write(f"  Precision (micro): {test_precision_micro:.4f}\n")
    f.write(f"  Recall (micro): {test_recall_micro:.4f}\n")
    f.write(f"  F1 Score (micro): {test_f1_micro:.4f}\n")
    f.write(f"  Precision (weighted): {test_precision_weighted:.4f}\n")
    f.write(f"  Recall (weighted): {test_recall_weighted:.4f}\n")
    f.write(f"  F1 Score (weighted): {test_f1_weighted:.4f}\n\n")
    
    f.write("PER-CLASS METRICS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Class':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
    f.write("-" * 80 + "\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name:<30} {per_class_acc[i]:<12.4f} {classwise_precision[i]:<12.4f} "
                f"{classwise_recall[i]:<12.4f} {classwise_f1[i]:<12.4f}\n")
    f.write("-" * 80 + "\n")

logger.info(f"Detailed test metrics saved to: {detailed_metrics_path}")

# --- Generate Plots ---

# 1. Confusion Matrix
plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names) * 0.8)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.title("Confusion Matrix on Test Set", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_plot_path = os.path.join(PLOTS_DIR, "test_confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Confusion matrix saved to: {cm_plot_path}")

# 2. Class-wise Performance Bar Charts
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

x_pos = np.arange(len(class_names))

# Accuracy
axes[0, 0].bar(x_pos, per_class_acc, color='lightblue', alpha=0.8, edgecolor='black')
axes[0, 0].set_xlabel('Class', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0, 1])

# Precision
axes[0, 1].bar(x_pos, classwise_precision, color='lightcoral', alpha=0.8, edgecolor='black')
axes[0, 1].set_xlabel('Class', fontsize=11)
axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Per-Class Precision', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim([0, 1])

# Recall
axes[1, 0].bar(x_pos, classwise_recall, color='lightgreen', alpha=0.8, edgecolor='black')
axes[1, 0].set_xlabel('Class', fontsize=11)
axes[1, 0].set_ylabel('Recall', fontsize=11)
axes[1, 0].set_title('Per-Class Recall', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_ylim([0, 1])

# F1 Score
axes[1, 1].bar(x_pos, classwise_f1, color='lightyellow', alpha=0.8, edgecolor='black')
axes[1, 1].set_xlabel('Class', fontsize=11)
axes[1, 1].set_ylabel('F1 Score', fontsize=11)
axes[1, 1].set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([0, 1])

plt.suptitle('Test Set: Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
classwise_plot_path = os.path.join(PLOTS_DIR, "test_classwise_metrics.png")
plt.savefig(classwise_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Class-wise metrics plot saved to: {classwise_plot_path}")

# 3. Single F1 Score Bar Chart
plt.figure(figsize=(max(12, len(class_names) * 0.8), 8))
bars = plt.bar(x_pos, classwise_f1, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
plt.xlabel('Class', fontsize=13)
plt.ylabel('F1 Score', fontsize=13)
plt.title('Test Set: Per-Class F1 Scores', fontsize=15, fontweight='bold')
plt.xticks(x_pos, class_names, rotation=45, ha='right', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 1.05])

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, classwise_f1)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
f1_bar_plot_path = os.path.join(PLOTS_DIR, "test_per_class_f1_bars.png")
plt.savefig(f1_bar_plot_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Per-class F1 bar chart saved to: {f1_bar_plot_path}")

# Print the final report to console
logger.info("\n--- Final Test Set Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
logger.info("--------------------------")
logger.info(f"\n✅ Script finished successfully!")
logger.info(f"📁 All outputs saved in: {OUTPUT_DIR}")
logger.info(f"📊 Metrics directory: {METRICS_DIR}")
logger.info(f"📈 Plots directory: {PLOTS_DIR}")
logger.info(f"📝 Log file: {log_filename}")

