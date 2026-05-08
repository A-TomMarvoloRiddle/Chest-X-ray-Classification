# Experiment 2 - Phase 3 (ResNet-18, Best Model)

## Overview

This folder contains the final Phase 3 results for Experiment 2, a transfer learning pipeline using `ResNet-18` pretrained on ImageNet for chest X-ray classification.

Task: binary classification of chest X-rays into:
- `normal`
- `pneumonia`

## Phase 3 Configuration

- Backbone: `torchvision.models.resnet18(weights=None)` with pretrained ImageNet weights loaded manually
- Final classifier: `Linear(512, 2)`
- Input size: `224 × 224`
- Data transforms:
  - `Resize((224, 224))`
  - `ToTensor()`
  - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- Augmentation: same as Phase 2
  - Random horizontal flips
  - Random rotations
  - `ColorJitter`
- Regularization:
  - Dropout
  - Weight decay
  - Label smoothing

## Best Model Files

- `outputs/best_model_phase3.pth`
- `outputs/metrics3.txt`
- `outputs/predictions3.csv`
- `outputs/sample_outputs/` — Grad-CAM sample outputs

## Performance
Accuracy : 0.9750
Precision: 0.9750
Recall : 0.9750
F1 : 0.9750
ROC-AUC : 0.9750

## Confusion Matrix:
[[78 2]
[ 2 78]]

This Phase 3 model is the strongest performer in Experiment 2.
