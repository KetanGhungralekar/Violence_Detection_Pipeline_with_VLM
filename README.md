# Violence Detection in Videos

## 🔍 Objective

This repository presents deep learning-based models for detecting violent content in video streams (real-time or pre-recorded). Multiple model architectures are explored and compared for their effectiveness on benchmark datasets.

---

## 📊 Datasets Used

1. **Real-Life Violence Situations Dataset** ([Kaggle](https://www.kaggle.com/datasets))
    - 2,000 video clips (1,000 violent, 1,000 non-violent)
    - Realistic: includes background noise, lighting variations, etc.

2. **A Dataset for Automatic Violence Detection in Videos** ([AIRT Lab](https://airt.ai/))
    - 350 annotated videos (230 violent, 120 non-violent)
    - Smaller, diverse, useful for benchmarking.

![Sample frame from dataset](vio_2.jpg)

---

## 🧠 Deep Learning Models Explored

### 1. YOLOv8
- **Approach:** Treated each frame independently; bounding box covers entire image.
- **Training:** 25 epochs, 224x224 images, used only on the 350-video dataset.
- **Performance:** mAP@50: 54%
    - **Limitation:** Not suitable due to lack of temporal understanding.

### 2. Video Masked Autoencoders (VMAE)
- **On 350-Video Dataset:**  
    - Frames: 16 sampled uniformly  
    - Accuracy: 88.57%  
    - Precision: 88%  
    - Recall: 95.65%  
    - F1-Score: 91.67%  
    - High recall → beneficial for real-world detection.
- **On 2000-Video Dataset:**  
    - Trained on 1,400 videos, then full set  
    - Accuracy: 95.96%  
    - Precision: 97.92%  
    - Recall: 94.00%  
    - F1-Score: 95.92%

### 3. 3D Convolutional Neural Networks (3D CNN)
- **On 2000-Video Dataset:**  
    - Pretrained I3D (MC3 18)  
    - Input shape: (3, 16, 112, 112)  
    - Accuracy: 97.75%
- **On 350-Video Dataset (ResNet3D-18):**  
    - Accuracy: 84.29%  
    - Precision: 95.35%  
    - Recall: 82.00%  
    - F1-Score: 88.17%

### 4. Video Vision Transformer (ViViT)
- **Pretrained:** google/vivit-b-16x2-kinetics400
- **On 2000-Video Dataset:**  
    - Accuracy: 99%  
    - Precision: 99.49%  
    - Recall: 98.5%  
    - F1-Score: 98.99%
- **On 350-Video Dataset:**  
    - Accuracy: 92.86%  
    - Precision: 95.56%  
    - Recall: 93.48%  
    - F1-Score: 94.51%

---

## 📈 Final Model Comparison

### On 2000-Video Dataset

| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| ViViT   | 99.00%   | 0.990     | 0.990  | 0.990    |
| 3D-CNN  | 97.75%   |   -       |   -    |    -     |
| VMAE    | 95.96%   | 0.9792    | 0.94   | 0.9592   |

### On 350-Video Dataset

| Model   | Accuracy | Precision | Recall   | F1-Score |
|---------|----------|-----------|----------|----------|
| ViViT   | 92.86%   | 0.920     | 0.930    | 0.920    |
| VMAE    | 88.57%   | 0.880     | 0.9565   | 0.9167   |
| 3D-CNN  | 84.29%   | 0.9535    | 0.820    | 0.8817   |

---

## 🔚 Conclusion

- **ViViT** is the most effective model, especially on larger datasets.
- **3D-CNN** demonstrates strong performance when more data is available.
- **VMAE** offers a balance between generalization and high recall, especially in low-data scenarios.
- **YOLOv8** is not suitable for violence detection in videos due to its inability to model temporal dependencies.

---

## 📁 Project Structure

```
Violence_Detection/
│
├── VMAE_code/                       # Video Masked Autoencoder code
├── ViVit_codes/                     # Video Vision Transformer code
├── Violence_Detection_I3D_Model/    # 3D-CNN (I3D) model code
├── README.md                        # Project information (this file)
└── Yolo_violence_detection.py       # YOLOv8 frame-based detection script
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch, torchvision
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- OpenCV
- Additional dependencies in `requirements.txt` (if available)

### Installation

```bash
git clone https://github.com/KetanGhungralekar/Violence_Detection.git
cd Violence_Detection
pip install -r requirements.txt
```

### Usage

1. **Prepare Data:**  
   - Download and extract datasets as described above.
   - Organize them in the appropriate directory structure as expected by each code folder.

2. **Training & Evaluation:**  
   - Execute the training scripts for each model available in their respective folders.
   - Example (ViViT):
     ```bash
     cd ViVit_codes
     python train_vivit.py --dataset_path /path/to/data
     ```
   - Evaluate model performance using provided evaluation scripts.


---

## 📧 Contact

For questions or collaborations, contact [Ketan Ghungralekar](https://github.com/KetanGhungralekar).
