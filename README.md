# Traffic Sign Classification Robustness

This project implements a robust traffic sign classification model using the GTSRB dataset and synthetic weather augmentations (Rain, Fog, Motion Blur).

## Key Results

The "Robust" model demonstrates a significant breakthrough in performance stability compared to the baseline trained on clean data:

| Condition | Baseline Accuracy | Robust Accuracy | Improvement |
|-----------|-------------------|-----------------|-------------|
| Clean     | 95.57%            | 97.19%          | +1.62%      |
| Rain      | 50.88%            | 91.79%          | +40.91%     |
| Fog       | 47.09%            | 83.71%          | +36.62%     |
| Blur      | 91.19%            | 94.96%          | +3.77%      |


## Setup
Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

All commands should be run from the project root directory.

### 1. Visualization
Visualize the synthetic weather augmentations:
```bash
python src/visualize_augmentations.py
```
Output: `augmentation_samples.png`

### 2. Training
**Train Baseline Model (Clean Data):**
```bash
python src/train.py --epochs 10
```
Saves to: `models/model_baseline.pth`

**Train Robust Model (Augmented Data):**
```bash
python src/train.py --augment --epochs 10
```
Saves to: `models/model_robust.pth`

### 3. Evaluation
Evaluate a trained model on Clean, Rain, Fog, and Blur test sets.

**Evaluate Baseline:**
```bash
python src/evaluate.py --model_path models/model_baseline.pth
```

**Evaluate Robust:**
```bash
python src/evaluate.py --model_path models/model_robust.pth
```

### 4. Interactive Demo
Launch the web-based demo to test the model with your own images and real-time weather effects:
```bash
python src/demo.py
```
Open the link provided in the terminal (usually `http://127.0.0.1:7860`).
