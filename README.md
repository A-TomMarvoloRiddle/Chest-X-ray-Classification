# Chest X-ray Classification Project

## Overview

This project implements a machine learning pipeline for classifying chest X-ray images into two categories: **Normal** and **Pneumonia**. The solution includes model training, evaluation, and interpretability analysis using Grad-CAM to highlight regions influencing predictions.

The project was developed to explore various DL architectures from scratch CNNs to pretrained models like ResNet-18 and DenseNet-121.

## Objective

- Classify chest X-ray images into Normal vs Pneumonia
- Analyze model predictions using Grad-CAM for interpretability
- Ensure the solution runs on CPU within time constraints
- Document experiments and iterative improvements

## Dataset

The dataset consists of chest X-ray images organized as follows:

```
dataset/
├── train/
│   ├── normal/      # Normal X-ray images
│   └── pneumonia/   # Pneumonia X-ray images
└── test/
    ├── normal/      # Test normal images
    └── pneumonia/   # Test pneumonia images
```

- **Download**: [dataset.zip](https://drive.google.com/file/d/1Lx47Vuqf2OXzGeDGAZMfno0TY8RQJB0M/view?usp=sharing)
- **Input Size**: Models use 224×224 grayscale images
- **Preprocessing**: Resize, normalization, and data augmentation

## Experiments

The project includes three main experiments with iterative improvements:

### Experiment 1: Baseline CNN from Scratch
- **Framework**: PyTorch on CPU
- **Architecture**: 4-block CNN (32→64→128→256 channels) with BatchNorm, Dropout
- **Augmentation**: Random horizontal flips, rotations (±10°), color jitter
- **Challenges**: Overfitting, Grad-CAM issues (fixed by proper hook registration)
- **Best Result**: Improved generalization but lower accuracy compared to transfer learning

### Experiment 2: ResNet-18 Pretrained
- **Framework**: PyTorch with torchvision
- **Architecture**: ResNet-18 pretrained on ImageNet, fine-tuned
- **Phases**:
  1. No augmentation, no regularization
  2. With augmentation (flips, rotations, color jitter)
  3. Augmentation + Dropout + Weight Decay + Label Smoothing
- **Grad-CAM**: Hooked on `layer4` for feature visualization

### Experiment 3: DenseNet-121 Pretrained
- **Framework**: PyTorch with torchvision
- **Architecture**: DenseNet-121 pretrained on ImageNet
- **Phases**:
  1. Frozen backbone, head-only training
  2. Unfrozen backbone + augmentation
  3. Full regularization (Dropout 0.3, WD 1e-4, LR 1e-4, Label Smoothing 0.1)

## Best Model

**Experiment 2, Phase 3 (ResNet-18 with regularization)** achieves the highest performance:

- **Accuracy**: 0.9750
- **Precision**: 0.9750
- **Recall**: 0.9750
- **F1-Score**: 0.9750
- **ROC-AUC**: 0.9750
- **Misclassifications**: 4 out of 160 test samples

This model balances performance with interpretability and computational efficiency.

## Project Structure

```
├── dataset/                 # Training and test data
├── evaluation/              # Evaluation scripts
├── Exp1/                    # Experiment 1: Baseline CNN
├── Exp2/                    # Experiment 2: ResNet-18
├── Exp3/                    # Experiment 3: DenseNet-121
├── Vibe/                    # Additional experiments through vibecoding (Ignore)
├── submission/              # Best Model
│   ├── outputs/             # Best model outputs
│   ├── experiments.md       # Detailed experiment log
│   └── requirements.txt     # Dependencies
├── experiments.md           # Experiment documentation
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation & Setup

1. **Clone/Download** the repository
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Dataset**: Extract `dataset.zip` to the project root
4. **Run Training**: Use the notebooks in `Exp2/` for the best model
5. **Evaluate**: Run evaluation scripts or notebooks

## Dependencies

- torch
- torchvision
- scikit-learn
- Pillow
- matplotlib
- numpy
- pandas
- pytorch_grad_cam
- opencv-python
- tqdm

## How to Run

### Training the Best Model
1. Navigate to `Exp2/`
2. Open `train_phase3.ipynb`
3. Run all cells to train ResNet-18 with regularization

### Testing & Evaluation
1. Use `test_phase3.ipynb` for inference
2. Check `outputs/metrics.txt` for performance metrics
3. View `predictions.csv` for test predictions

### Model Interpretation
- Grad-CAM heatmaps are generated automatically
- Sample outputs in `submission/outputs/sample_outputs/`

## Results & Outputs

- **Metrics**: Accuracy, precision, recall, F1, ROC-AUC
- **Predictions**: CSV file with image names and predicted labels
- **Interpretability**: Grad-CAM overlays showing influential regions
- **Sample Outputs**: 3-5 example images with heatmaps

## Key Learnings

- Transfer learning significantly outperforms from-scratch CNNs on small medical datasets
- Data augmentation and regularization are crucial for generalization
- Grad-CAM provides valuable insights into model decision-making
- Proper hook registration is essential for accurate feature visualization
- Iterative experimentation leads to better understanding and results

## Links

- [GitLab Repository](https://gitlab.com/A-TomMarvoloRiddle/ScanO-assign-Apaar)
- [Dataset Download](https://drive.google.com/file/d/1Lx47Vuqf2OXzGeDGAZMfno0TY8RQJB0M/view?usp=sharing)

