import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- 1. CONFIGURATION ---
# !!! Make sure these match your main training script !!!
OUTPUT_DIR = "./vivit-finetuned/checkpoint-3500" 
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Path to the trainer state file in your main output directory
STATE_FILE_PATH = os.path.join(OUTPUT_DIR, "trainer_state.json")

# Create output directories if they don't exist
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Loading history from: {STATE_FILE_PATH}")

# --- 2. LOAD HISTORY ---
try:
    with open(STATE_FILE_PATH, 'r') as f:
        trainer_state = json.load(f)
    history = trainer_state["log_history"]
    print(f"Successfully loaded log history with {len(history)} entries.")
except FileNotFoundError:
    print(f"ERROR: 'trainer_state.json' not found at {STATE_FILE_PATH}", file=sys.stderr)
    print("Please check your OUTPUT_DIR path.", file=sys.stderr)
    sys.exit(1)
except KeyError:
    print(f"ERROR: 'log_history' key not found in {STATE_FILE_PATH}.", file=sys.stderr)
    print("The file might be corrupted or from a different trainer version.", file=sys.stderr)
    sys.exit(1)

# --- 3. GENERATE PLOTS AND CSV (This is the fixed 'Step 7a' and '7b') ---

print("Generating reports and plots...")

# --- 7a. Save Epoch-wise Metrics ---

# Convert history to DataFrame
df_history = pd.DataFrame(history)

# Check if history is empty or missing data
if "loss" not in df_history.columns and "eval_loss" not in df_history.columns:
    print("ERROR: No 'loss' or 'eval_loss' columns found in log history.", file=sys.stderr)
    print("No metrics to plot.", file=sys.stderr)
    sys.exit(1)

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


# --- 7b. Plot Epoch-wise Metrics ---

# Plot Loss
if "train_loss" in df_epoch_metrics.columns and "val_loss" in df_epoch_metrics.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["train_loss"], label="Train Loss", marker='o')
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "loss_vs_epoch.png"))
    print(f"Loss plot saved to: {PLOTS_DIR}/loss_vs_epoch.png")
else:
    print("Skipping loss plot (missing train or validation loss data).")

# Plot F1-Score
if "val_f1" in df_epoch_metrics.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_f1"], label="Validation F1 (Weighted)", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score (Weighted)")
    plt.title("Validation F1 Score Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "f1_vs_epoch.png"))
    print(f"F1 plot saved to: {PLOTS_DIR}/f1_vs_epoch.png")
else:
    print("Skipping F1 plot (missing 'val_f1' data).")

# Plot Accuracy
if "val_accuracy" in df_epoch_metrics.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df_epoch_metrics["epoch"], df_epoch_metrics["val_accuracy"], label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_epoch.png"))
    print(f"Accuracy plot saved to: {PLOTS_DIR}/accuracy_vs_epoch.png")
else:
    print("Skipping accuracy plot (missing 'val_accuracy' data).")

print("\n--- Plotting complete ---")
