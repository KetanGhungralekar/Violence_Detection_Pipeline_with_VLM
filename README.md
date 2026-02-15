# Violence Detection Pipeline with Vision Language Models (VLM)

## 🔍 Objective

This repository presents a comprehensive violence detection pipeline that combines deep learning-based video classification models with Vision Language Models (VLMs) for enhanced understanding and detection of violent content in videos. Multiple model architectures are explored and compared for their effectiveness on benchmark datasets.

---

## 📊 Datasets Used

1. **UCF Crime Dataset (UCF_MERGED_NORM)**
    - Merged and normalized dataset for multi-class violence detection
    - Contains Normal and Abnormal classes with fine-grained categories
    - Used for training ViViT and I3D models with multi-task learning

2. **Real-Life Violence Situations Dataset**
    - 2,000 video clips (1,000 violent, 1,000 non-violent)
    - Binary classification: Violence vs Non-Violence
    - Used for VMAE and 3D-CNN models

3. **Smaller Benchmark Dataset**
    - 350 annotated videos (230 violent, 120 non-violent)
    - Used for model evaluation and benchmarking

---

## 🧠 Deep Learning Models Explored

### 1. Video Vision Transformer (ViViT)
- **Implementation:** `ViViT_UCF_Merged/` directory
- **Pretrained Model:** google/vivit-b-16x2-kinetics400
- **Training Scripts:**
  - `ViViT_UCF_Merged/hf_vivit.py` - HuggingFace Transformers-based training
  - `ViViT_UCF_Merged/prev_model/kinetics.py` - Custom implementation with multi-task learning
  - `ViViT_UCF_Merged/prev_model/vivit.py` - ViViT model architecture
  - `ViViT_UCF_Merged/prev_model/module.py` - Building blocks for ViViT
- **Features:**
  - Multi-task learning (binary + fine-grained classification)
  - 32 frames per video @ 224x224 resolution
  - Binary and multi-class classification heads
- **Data Processing:**
  - `ViViT_UCF_Merged/video_clips.py` - Generate clips from videos
  - `ViViT_UCF_Merged/Normalisation.py` - Normalize video clips
  - `ViViT_UCF_Merged/extract_augment.py` - Data augmentation
- **Performance:**
  - Accuracy: 99% on large dataset
  - Best overall model with highest precision and recall

### 2. Video Masked Autoencoders (VMAE)
- **Implementation:** 
  - `vmae_mod.py` - Training script with multi-task learning
  - `vame_infer_v.py` - Inference script
- **Pretrained Model:** MCG-NJU/videomae-base
- **Configuration:**
  - 16 frames sampled uniformly @ 224x224
  - Batch size: 8, Epochs: 200, Patience: 20
  - Learning rate: 1e-4 with ReduceLROnPlateau scheduler
  - Mixed precision training (AMP)
- **Features:**
  - Dual-head architecture (binary + fine-grained classification)
  - Label smoothing for better generalization
  - Early stopping with validation monitoring
- **Checkpoint:** `best_vmae_16f224_updated_binary.pth`
- **Splits:** `splits_vaibhav.json` (train/inner_val/val)
- **Performance:**
  - Accuracy: 95.96% on 2000-video dataset
  - High recall beneficial for real-world detection

### 3. 3D Convolutional Neural Networks (I3D - MC3_18)
- **Implementation:**
  - `i3d_Multitask_model_train.py` - Training with multi-task heads
  - `i3d_Multitask_model_test.py` - Evaluation and testing
- **Architecture:** 
  - Backbone: MC3-18 pretrained on Kinetics
  - Binary head: Linear(512 → 256 → 2) with ReLU
  - Fine-grained head: Linear(512 → 512 → num_classes) with ReLU
- **Configuration:**
  - 16 frames @ 224x224 resolution
  - Input shape transformation: (B, T, C, H, W) → (B, C, T, H, W)
  - Batch size: 8, Epochs: 200
  - AdamW optimizer with weight decay: 1e-3
  - Label smoothing: 0.1
- **Checkpoints:** 
  - `checkpoints/best_i3d_multitask.pth` (best validation)
  - `final_i3d_multitask.pth` (final model)
- **Splits:** `splits.json` (80/10/10 train/val/test)
- **Output:** 
  - `test_predictions.csv` - Detailed predictions
  - `test_summary.json` - Confusion matrices and metrics
- **Performance:**
  - Accuracy: 97.75% on 2000-video dataset
  - Strong performance with larger datasets

### 4. Vision Language Models (VLM)
- **Implementation:** `NLP/` directory
- **Model:** Qwen-2VL for video understanding
- **Notebooks:**
  - `qwen-2vl-inference_with_modified_prompts.ipynb` - Inference with custom prompts
- **Features:**
  - Enhanced context understanding through natural language
  - Ability to explain and describe violence in videos
  - Multimodal analysis combining vision and language
- **Outputs:**
  - `VLM_Output_With_Original_Prompt.pdf`
  - `VLM_Output_With_Modified_Prompt.pdf`

---

## 📈 Model Performance Summary

### Binary Violence Classification

| Model   | Accuracy | Precision | Recall | F1-Score | Frames | Notes |
|---------|----------|-----------|--------|----------|--------|-------|
| ViViT   | 99.00%   | 99.49%    | 98.50% | 98.99%   | 32     | Best overall performance |
| I3D (MC3-18) | 97.75% | -      | -      | -        | 16     | Strong with large datasets |
| VMAE    | 95.96%   | 97.92%    | 94.00% | 95.92%   | 16     | High recall, good generalization |

---

## 🔚 Conclusion

- **ViViT** achieves the highest accuracy (99%) and is the most effective model overall, especially when combined with multi-task learning for both binary and fine-grained classification.
- **I3D (MC3-18)** demonstrates strong performance (97.75%) and is computationally efficient with 3D convolutions, making it suitable for deployment scenarios.
- **VMAE** offers excellent balance (95.96%) between generalization and high recall, especially beneficial in low-data scenarios and real-world detection where false negatives are costly.
- **VLM Integration** adds interpretability by providing natural language explanations of violence detection, enhancing the pipeline's trustworthiness and usability.
- **Multi-task Learning** approach with both binary and fine-grained classification heads improves model robustness and provides more detailed analysis of violence types.

---

## 📁 Project Structure

```
Violence_Detection_Pipeline_with_VLM/
│
├── ViViT_UCF_Merged/              # Video Vision Transformer implementation
│   ├── hf_vivit.py                # HuggingFace-based ViViT training
│   ├── video_clips.py             # Generate video clips
│   ├── Normalisation.py           # Video normalization
│   ├── extract_augment.py         # Data augmentation
│   ├── video_analyzer.py          # Analyze video statistics
│   ├── plot_metrics.py            # Visualization utilities
│   ├── prev_model/                # Custom ViViT implementation
│   │   ├── kinetics.py            # Training/inference script
│   │   ├── vivit.py               # ViViT architecture
│   │   ├── vivit_multitask.py     # Multi-task ViViT
│   │   ├── module.py              # Model building blocks
│   │   ├── binary_class/          # Binary classification version
│   │   └── normal_class/          # Normal classification version
│   └── README.md                  # ViViT-specific documentation
│
├── NLP/                           # Vision Language Model integration
│   ├── qwen-2vl-inference_with_modified_prompts.ipynb
│   ├── VLM_Output_With_Original_Prompt.pdf
│   ├── VLM_Output_With_Modified_Prompt.pdf
│   └── Videos/                    # Sample videos for VLM testing
│
├── vmae_mod.py                    # VMAE training with multi-task learning
├── vame_infer_v.py                # VMAE inference script
├── i3d_Multitask_model_train.py   # I3D MC3-18 training
├── i3d_Multitask_model_test.py    # I3D evaluation and testing
│
├── splits.json                    # I3D dataset splits (80/10/10)
├── splits_vaibhav.json            # VMAE dataset splits
├── checkpoints/                   # Saved model checkpoints
│   └── best_i3d_multitask.pth
├── best_vmae_16f224_updated_binary.pth  # VMAE checkpoint
│
└── README.md                      # Main documentation (this file)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- HuggingFace Transformers
- OpenCV (cv2)
- NumPy, Pandas
- scikit-learn
- tqdm

### Installation

```bash
git clone https://github.com/KetanGhungralekar/Violence_Detection_Pipeline_with_VLM.git
cd Violence_Detection_Pipeline_with_VLM
pip install torch torchvision transformers opencv-python numpy pandas scikit-learn tqdm
```

### Dataset Preparation

Organize your dataset in the following structure:
```
UCF_MERGED_NORM/
├── Normal/
│   ├── Class1/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── Class2/
└── Abnormal/
    ├── Violence1/
    └── Violence2/
```

### Training Models

#### 1. Train VMAE Model

```bash
# Training (creates splits_vaibhav.json on first run)
python vmae_mod.py

# Resume training from checkpoint
python vmae_mod.py --resume

# Inference
python vame_infer_v.py
```

**Configuration (in `vmae_mod.py`):**
- `DATA_ROOT`: Path to UCF_MERGED_NORM dataset
- `BATCH_SIZE`: 8 (default)
- `EPOCHS`: 200
- `PATIENCE`: 20 (early stopping)
- `FRAMES_PER_VIDEO`: 16
- `IMAGE_SIZE`: 224

#### 2. Train I3D Model

```bash
# Training
python i3d_Multitask_model_train.py

# Testing
python i3d_Multitask_model_test.py
```

**Outputs:**
- `checkpoints/best_i3d_multitask.pth` - Best model
- `test_predictions.csv` - Detailed predictions
- `test_summary.json` - Metrics and confusion matrix

#### 3. Train ViViT Model

```bash
cd ViViT_UCF_Merged

# Using HuggingFace Transformers
python hf_vivit.py

# Using custom implementation
cd prev_model
python kinetics.py
```

**Configuration (in `hf_vivit.py`):**
- `DATASET_PATH`: Path to video clips
- `NUM_FRAMES`: 32
- `NUM_CLASSES`: Adjust based on your dataset
- `MAX_EPOCHS`: 200
- `EARLY_STOPPING_PATIENCE`: 20

### Data Processing for ViViT

```bash
cd ViViT_UCF_Merged

# Generate video clips
python video_clips.py

# Normalize clips
python Normalisation.py

# Augment data
python extract_augment.py

# Analyze dataset
python video_analyzer.py
```

### Using Vision Language Models

Open and run the Jupyter notebook:
```bash
cd NLP
jupyter notebook qwen-2vl-inference_with_modified_prompts.ipynb
```

### Model Inference

Each model provides inference capabilities:

**VMAE:**
```python
# Uses best_vmae_16f224_updated_binary.pth
# Outputs: predictions_test.csv
python vame_infer_v.py
```

**I3D:**
```python
# Uses checkpoints/best_i3d_multitask.pth
# Outputs: test_predictions.csv, test_summary.json
python i3d_Multitask_model_test.py
```

**ViViT:**
```bash
cd ViViT_UCF_Merged/prev_model
# Configure inference in kinetics.py
python kinetics.py
```


---

## 📧 Contact

For questions or collaborations, contact [Ketan Ghungralekar](https://github.com/KetanGhungralekar).
