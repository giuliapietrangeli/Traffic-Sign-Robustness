# Weather-Robust Traffic Sign Recognition 

This repository implements a robust deep learning pipeline for traffic sign classification, specifically designed to maintain high performance under adverse weather conditions. By leveraging the German Traffic Sign Recognition Benchmark (GTSRB) and advanced synthetic data augmentation, this project addresses the reliability gap in autonomous driving perception systems.

## Project Overview

Standard convolutional neural networks (CNNs) often suffer from significant performance degradation when exposed to environmental corruptions such as rain, fog, or motion blur. This project demonstrates that a specialized training strategy using synthetic weather artifacts can bridge this domain gap, achieving a substantial increase in classification accuracy without compromising performance on clean data.

### Technical Methodology

1. **Modified ResNet18 Architecture**: The standard ResNet18 model was adapted to process 32x32 pixel input images. The initial layers (stride and max-pooling) were modified to preserve spatial resolution, which is critical for identifying the fine-grained features of small-scale traffic signage.
2. **Synthetic Data Augmentation**: Utilizing the Albumentations library, the training pipeline incorporates dynamic simulation of weather effects, including:
   - **Rain**: Simulated streaks and droplets (RandomRain).
   - **Fog**: Atmospheric haze and contrast reduction (RandomFog).
   - **Motion Blur**: Simulated sensor or object movement (MotionBlur).
3. **Robust Training**: The model was trained using a 50% probability of applying these transformations, forcing the network to learn noise-invariant features.

## Quantitative Results

The "Robust" model exhibits a remarkable improvement in stability compared to the baseline model trained exclusively on clean data:

| Condition | Baseline Accuracy | Robust Accuracy | Improvement |
|-----------|-------------------|-----------------|-------------|
| Clean     | 95.57%            | 97.19%          | +1.62%      |
| Rain      | 50.88%            | 91.79%          | +40.91%     |
| Fog       | 47.09%            | 83.71%          | +36.62%     |
| Blur      | 91.19%            | 94.96%          | +3.77%      |

## Repository Structure

* `src/`: Core source code containing training logic, evaluation scripts, and visualization utilities.
* `models/`: Pre-trained model checkpoints (.pth) for both the baseline and robust architectures.
* `Report.pdf`: Comprehensive technical report detailing the experimental setup and analysis.
* `demo.py`: Interactive web-based interface for real-time testing and visualization.

## Setup and Usage

### Installation
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Training
To train the robust model with synthetic weather augmentation:
```bash
python src/train.py --augment --epochs 10
```

### Evaluation
Evaluate the model performance across different corrupted test sets:
```bash
python src/evaluate.py --model_path models/model_robust.pth
```

### Interactive Demo
Launch the interactive demo to test the model with custom images and simulated weather effects:
```bash
python src/demo.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed by Giulia Pietrangeli and Lorenzo Musso.*
