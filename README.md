# ML Assignment — Chest X-ray Classification

## Objective

Build a machine learning pipeline to classify chest X-ray images into:

- Normal
- Pneumonia

Additionally, analyze which regions of the image influence your model’s predictions.

---

## Dataset

- Download: dataset.zip(https://drive.google.com/file/d/1Lx47Vuqf2OXzGeDGAZMfno0TY8RQJB0M/view?usp=sharing)
- After download, extract: unzip dataset.zip 

Structure:

```
dataset/   
├── train/    
│   ├── normal/    
│   └── pneumonia/  
└── test/    
    ├── normal/    
    └── pneumonia/
```

---

## Tasks

### 1. Data Handling

- Load and process images
- You may choose how to handle varying image sizes

---

### 2. Model Training

- Train any classification model
- Any framework allowed

---

### 3. Evaluation

- Evaluate on test set
- Report accuracy

---

### 4. Model Interpretation (Important)

- Show which regions influenced predictions
- Examples:
  - Grad-CAM
  - Heatmaps

---

## Constraints

- Your solution must run on CPU
- Training + inference must complete within 60 minutes
- Keep your solution simple and explainable

---

## Deliverables

Submit a .zip:

```
submission/   
├── README.md  
├── requirements.txt  
├── train.py OR notebook.ipynb 
├── experiments/
   └──  experiments.md
├── outputs/    
   ├── metrics.txt    
   ├── predictions.csv    
   └── sample_outputs/
```

---

## Experiment Log (Required)

You must include a file:

```
experiments/experiments.md
```

Document the different approaches you tried.

### For each experiment, include:

- What you changed (model, input size, preprocessing, etc.)
- Why you tried it
- What happened (result / observation)
- What you learned

### Example Format

```
#### Experiment 1 — Baseline
- Used simple CNN with default settings  
- Input parameters: <input size, batch size, learning rate, etc.> 
- Result: Low accuracy (~65%)  
- Observation: Model overfitting

---

#### Experiment 2 — Transfer Learning
- Used pretrained ResNet18  
- Same input parameters
- Result: Significant improvement (~80%)  
- Observation: Transfer learning works better on small dataset  

---

#### Experiment 3 — Data Augmentation
- Added random flips and rotations  
- Result: Slight improvement  
- Observation: Helps generalization  

---
```

### Expectations

- At least 2-3 experiments
- Clear progression of ideas
- Honest reporting (including failures)

**This section is important—we evaluate how you think and iterate, not just your final model.**

### predictions.csv format

```
image_name,label 
img_001.png,normal
```

---

### sample_outputs/

Include:

- 3–5 images with model interpretation overlays

---

## Important

- You may use AI tools
- You must understand and explain your solution

---

## Evaluation Criteria

- Code quality
- Clarity of explanation
- Data handling decisions
- Model interpretation
- Reasoning

Accuracy alone is not the primary metric

---

## Handy Tips

- Models require fixed-size inputs — decide how you handle this
- Start simple before optimizing
- Inspect sample images before training
- Consider dataset imperfections
- Focus on explainability

