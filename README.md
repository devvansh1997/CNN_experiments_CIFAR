# CIFAR-10 CNN Experiments

This repository contains a complete training pipeline for experimenting with different CNN architectures on the CIFAR-10 dataset.  
The focus of the project is to study **how network depth and architectural choices affect accuracy** using three models:

- **CNN_Small** — 3 conv layers  
- **CNN_Large** — 6 conv layers  
- **CNN_XL** — 6 conv layers + BatchNorm + higher channel widths  

All training runs were executed for **30 epochs** using **Cross-Entropy Loss + SGD (momentum = 0.9)**.  
Weights & Biases (W&B) was used to track losses, accuracy curves, and parameter statistics.

---

## Features
- Modular design (models, utils, scripts, config)
- Toggle between models via YAML config
- Automatic logging to Weights & Biases
- Configurable hyperparameters (LR, batch size, scheduler, weight decay)
- Saves best model checkpoint automatically
- Supports CPU, CUDA, and Apple MPS devices

---

## Results Summary

### Accuracy Comparison (30 epochs)

| Model      | Train Acc | Test Acc | Params (approx.) |
|------------|-----------|----------|-------------------|
| CNN_Small  | 81.83%    | 80.50%   | ~620K             |
| CNN_Large  | 82.91%    | 82.16%   | ~1.7M             |
| CNN_XL     | **94.12%** | **89.42%** | ~3.4M             |

**Key Insight:**  
Increasing model depth helps, but **BatchNorm + larger feature maps** produces the biggest jump in performance.

---

## Training Curves

Training & testing plots (saved under `/outputs/plots/`):

- `train_acc.png`
- `test_acc.png`
- `train_loss.png`
- `test_loss.png`

All runs are also logged on W&B with interactive graphs.

---

## ⚙️ Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Configure the experiment
Edit config/default.yaml:
```yaml
experiment_name: "cnn_experiments_cifar"
run_name: "cnn_xl"
model:
  type: "cnn_xl"   # cnn_small, cnn_large, or cnn_xl
```
### 3. Run training
```bash
python main.py
```

## Models Overview
### CNN_Small
- 3 convolutional blocks (Conv → ReLU → MaxPool)
- 2 fully connected layers
- Lightweight baseline
### CNN_Large
- 6 convolutional layers arranged in 3 blocks
- More feature extraction capacity
- Higher accuracy than small model
### CNN_XL
- Same structure as Large but replaces plain conv stacks with:
- Higher channel sizes (64/128/256)
- BatchNorm in every conv layer
- Most stable gradients and best performance

## Outputs
After training, the following are generated:
- outputs/checkpoints/<model>_best.pth
- wandb/ run logs
- outputs/plots/*.png

## Report
The LaTeX report includes:
- Model architectures
- Training methodology
- Tables comparing accuracy
- Plots of losses & accuracies
- Discussion on depth, BatchNorm effects, and performance

## Acknowledgments
- Dataset from CIFAR-10 (Krizhevsky et al.)
- Experiment tracking via Weights & Biases
- Models implemented using PyTorch 
