## Detailed work can be viewed here: [Apaar ScanO Assig. on GitLab](https://gitlab.com/A-TomMarvoloRiddle/ScanO-assign-Apaar)

### Experiment 1 - Baseline CNN from Scratch
> Framework: PyTorch | Device: CPU
> Task: Normal vs Pneumonia | Input: 224×224 grayscale X-rays

1. Took 3 block CNN & 224x224 input - Result: Overfits quickly.
2. Then added Random Horizontal Filps, Random Rotations (10), ColorJitter(brightness=0.2, contrast=0.2) - Result: Recall for normal class improved, Both classes improved in precision. 
3. Then added BatchNormalization (2d) after each Conv layer - 4 ConvBlocks (32 -> 64 -> 128 -> 256 channels) - Result: Smoother loss curve
4. Removed 4th Conv.Block - to see if fewer parameters = less overfitting + fast training on CPU - Result: Worse then 4 Conv. Block. but training time dropped
5. Reduced input size to 128x128 - Result: Faster training but Grad-CAM quality degrades

Best Config = 4-block CNN + BatchNorm + Dropout + Augmentation + 224×224

- Major Challenge -> solid blue heatmap (CAM values are all zero or near-zero after ReLU) even in best config
- The Fix -:
1. Replaced self.features = nn.Sequential(block1..4) with 4 explicit named attributes self.block1 … self.block4 -> Makes model.block4 a directly hookable module, not buried inside a Sequential
2. Hook is now model.block4.register_forward_hook(...) -> Captures full (B, 256, 14, 14) spatial maps before GAP runs
3. input_tensor.requires_grad_(True) added in generate() -> Ensures gradient flows all the way back to the hooked layer even in eval() mode

I observed that even after tweaking the no. of epochs, learning rate, dropouts, & doing augmentation --- the Base CNN model is not performing well --- can be clearly seen in the results that model is not focusing on the correct parts of the xray, the accuracy may be a outcome of biaseness of the dataset

### Experiment 2 - ResNet-18 pretrained on ImageNet

> I implemented this in 3 phases to bypass the chaos of updating the same code structure of diff. experimentations.

Chose ResNet-18 pretrained on ImageNet coz it's lightweight, stable, CPU-friendly transfer-learning baseline with strong feature extraction and excellent Grad-CAM interpretability.

- 3 Phases (3 separate saved models + metrics):
1. Augmentation: None (Resize + Normalize only) 
    Regularisation: None
2. Augmentation: like HFlip + Rotation + ColorJitter, etc. 
    Regularisation: None 
3. Augmentation: Same as Phase 2 
    Regularisation: Dropout, Optimizer( Learning rate & Weight Decay), Label Smoothing

- Grad-CAM: Hooked on model.layer4 — last conv block of ResNet-18, outputs (B, 512, 7, 7) before avgpool

- Phase 3 performed better than Phase 1 & Phase 2 (can be seen through metrics of both)

### Experiment 3 - DenseNet121 (pretrained on ImageNet) 

> I implemented this in 3 phases to bypass the chaos of updating the same code structure of diff. experimentations.

Chose DenseNet121 (pretrained on ImageNet) coz it's dense feature reuse helps capture subtle medical-image patterns efficiently, making it strong for chest X-ray classification while remaining computationally manageable.


| Phase | Backbone | Augmentation | Dropout | WD   | LR   | Label Smooth | Notes                          |
|-------|----------|-------------|---------|------|------|--------------|--------------------------------|
| 1     | Frozen   | None        | None    | None | 1e-3 | None         | Head-only baseline             |
| 2     | Unfrozen | Flip+Rot+CJ | None   | None | 1e-3 | None         | Full fine-tune + augmentation  |
| 3     | Unfrozen | Flip+Rot+CJ | 0.3    | 1e-4 | 1e-4 | 0.1          | Regularized fine-tuning        |


### Phase Phase 3 of Exp2 Is the Best

Top Accuracy: 0.9750 surpasses all others, with perfect balance in precision, recall, F1, and ROC-AUC (0.9750). The confusion matrix shows only 4 misclassifications out of 160 samples.

Comparison: This edges out Exp2 Phase 1 (0.9688) and significantly outperforms the others.