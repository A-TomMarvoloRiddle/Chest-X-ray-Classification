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

The project follows a structured organization with experiment isolation, clear deliverables, and comprehensive documentation:

### Root Level Files
- **`experiments.md`**: Comprehensive log documenting all experiments conducted, including changes made, results, observations, and learnings from each iteration across Experiments 1-3
- **`README.md`**: Project documentation with overview, setup instructions, experiment summaries, and results
- **`requirements.txt`**: Python dependencies required to run the project (torch, torchvision, scikit-learn, etc.)

### Main Directories

#### **`dataset/`**
Contains the chest X-ray image dataset:
- **`train/`**: Training images
  - **`normal/`**: Normal chest X-rays for training
  - **`pneumonia/`**: Pneumonia chest X-rays for training
- **`test/`**: Test images
  - **`normal/`**: Normal chest X-rays for testing
  - **`pneumonia/`**: Pneumonia chest X-rays for testing

#### **`evaluation/`**
- **`evaluation.py`**: Scripts for model evaluation, metrics calculation, and performance analysis

#### **`Exp1/`** (Experiment 1: Baseline CNN from Scratch)
- **`requirements.txt`**: Experiment-specific dependencies
- **`test1.ipynb`**, **`test2.ipynb`**: Testing notebooks for different configurations
- **`train1.ipynb`**, **`train2.ipynb`**: Training notebooks for baseline CNN experiments
- **`outputs/`**: Results from Exp1
  - **`best_model.pth`**: Best trained CNN model
  - **`metrics.txt`**: Performance metrics
  - **`predictions.csv`**: Test set predictions
  - **`sample_outputs/`**: Grad-CAM visualization examples

#### **`Exp2/`** (Experiment 2: ResNet-18 Pretrained)
- **`requirements.txt`**: Dependencies for ResNet experiments
- **`test_phase1.ipynb`**, **`test_phase2.ipynb`**, **`test_phase3.ipynb`**: Testing notebooks for each phase
- **`train_phase1.ipynb`**, **`train_phase2.ipynb`**, **`train_phase3.ipynb`**: Training notebooks for ResNet-18 phases
- **`submission/`**: Contains final outputs for this experiment
  - **`outputs/`**: Best models and results from all phases
    - **`best_model_phase1.pth`**, **`best_model_phase2.pth`**, **`best_model_phase3.pth`**: Saved models
    - **`metrics1.txt`**, **`metrics2.txt`**, **`metrics3.txt`**: Performance metrics for each phase
    - **`predictions1.csv`**, **`predictions2.csv`**, **`predictions3.csv`**: Predictions
    - **`sample_outputs/`**: Grad-CAM visualizations

#### **`Exp3/`** (Experiment 3: DenseNet-121 Pretrained)
- **`test.ipynb`**: Testing notebook
- **`train.ipynb`**: Training notebook
- **`outputs/`**: Results from DenseNet experiments
  - **`best_model_phase1.pth`**, **`best_model_phase2.pth`**, **`best_model_phase3.pth`**: DenseNet models
  - **`metrics.txt`**: Combined metrics
  - **`predictions_phase1.csv`**, **`predictions_phase2.csv`**, **`predictions_phase3.csv`**: Phase-specific predictions
  - **`predictions_phase1_test.csv`**, **`predictions_phase2_test.csv`**, **`predictions_phase3_test.csv`**: Test predictions
  - **`sample_outputs/`**: Visualization folders for each phase
    - **`phase1/`**, **`phase1_test/`**, **`phase2/`**, **`phase2_test/`**, **`phase3/`**, **`phase3_test/`**

#### **`Vibe/`** (Unstructured Vibe Coding Exploratory Experiments)
- **`exp1.md`**, **`exp2.md`**: Experiment notes
- **`Exp1.py`**: Python script for experiments
- **`Exp2.ipynb`**: Jupyter notebook for experiments
- **`try1.ipynb`**, **`try1-1.ipynb`**: Trial notebooks
- **`experiments/`**: Experiment documentation and outputs
  - **`experiments.md`**: Experiment log
  - **`outputs/`**: Various model outputs
    - **`baseline_cnn.pt`**, **`best_resnet18.pth`**: Saved models
    - **`metrics.txt`**, **`predictions.csv`**: Metrics and predictions
    - **`exp1/`**, **`exp2/`**: Sub-experiment results with sample outputs
    - **`sample_outputs/`**: General visualization examples

#### **`submission/`** (Final Deliverables)
This folder contains the polished, submission-ready version of the best work:
- **`README.md`**: Submission README
- **`requirements.txt`**: Dependencies
- **`test_phase1_Exp2.ipynb`**, **`train_phase1_Exp2.ipynb`**: Notebooks for the best model (Exp2 Phase 1)
- **`experiments/`**: 
  - **`experiments.md`**: Experiment documentation
- **`outputs/`**: Final model outputs
  - **`best_model_phase3.pth`**: Best performing model (ResNet-18 Phase 3)
  - **`metrics3.txt`**: Final metrics (97.5% accuracy)
  - **`predictions3.csv`**: Final predictions
  - **`sample_outputs/`**: Grad-CAM examples for submission

### Organization Philosophy
- **Experiment Isolation**: Each major experiment (Exp1, Exp2, Exp3) has its own folder with complete code, data, and outputs
- **Phase Separation**: Within experiments, different configurations are separated into phases
- **Deliverables**: The `submission/` folder contains the clean, final version for evaluation
- **Documentation**: Experiment logs and READMEs provide context for each component
- **Outputs**: Consistent structure across experiments (metrics.txt, predictions.csv, sample_outputs/)

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

