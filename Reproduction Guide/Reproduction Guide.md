# DFM-KM MPC Reproduction (Essential Codebase)

This folder contains the essential code and checkpoints to reproduce the DFM-KM MPC evaluation.

## 1. Setup

### 1.1 Data
You need to link the `traffic-data` folder here.
```bash
cd pytorch-PPUU-ICLR2022
ln -s /path/to/your/traffic-data traffic-data
```
*(Replace `/path/to/your/traffic-data` with the actual path to your processed data, e.g., `../../../../traffic-data`)*

### 1.2 Install
```bash
pip install .
pip install torchvision submitit wandb imageio
```

## 2. Run Evaluation

### 2.1 MPC Evaluation (Braking Fix)
```bash
./run_viz.sh
```
Results will be in `results/mpc_eval_viz/visualizer/videos/`.

### 2.2 Prediction Evaluation
```bash
PYTHONPATH=. python ppuu/eval_prediction.py \
    --values dataset="traffic-data/processed/data_i80_v0" \
             forward_model_path="results/fm/fm/seed=42_9/checkpoints/last.ckpt" \
             output_dir="results/prediction_eval"
```
Results will be in `results/prediction_eval/prediction_eval.json`.
