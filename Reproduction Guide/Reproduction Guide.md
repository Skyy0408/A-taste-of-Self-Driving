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

## 3. Key Programs & Modifications

Here is a summary of the important programs and the modifications made to the original codebase:

### 3.1 `ppuu/eval_prediction.py` (New)
*   **Purpose**: Evaluates the Forward Model's prediction accuracy (ADE/FDE) on the test set.
*   **Modifications**:
    *   **Standalone Evaluation**: Created a dedicated script to evaluate prediction performance independent of the policy.
    *   **Metrics**: Implemented `DisplacementErrorMetric` to calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).
    *   **Device Compatibility**: Added support for CPU, MPS (Mac), and CUDA devices to ensure it runs on various hardware.
    *   **Open-Loop Testing**: Configured to use Ground Truth actions to purely test the model's dynamics prediction capabilities.

### 3.2 `ppuu/modeling/policy/mpc.py` (Modified)
*   **Purpose**: Implements the Model Predictive Control (MPC) policy using the Cross-Entropy Method (CEM).
*   **Modifications**:
    *   **Uncertainty Visualization**: Added logic in `_ce_optimize` to unfold multiple future trajectories (samples) for the selected best action sequence. This allows visualizing the model's uncertainty (e.g., showing multiple possible future paths) in the generated videos.

### 3.3 `run_viz.sh` (New)
*   **Purpose**: Shell script to streamline the MPC evaluation with visualization.
*   **Modifications**:
    *   **Automation**: Sets up the environment (`PYTHONPATH`) and manages output directories.
    *   **Visualization Config**: Configures the evaluator to generate overlays (ego-view, road, lanes) and cost landscapes (`cost.build_overlay=True`, `cost.build_cost_profile_and_traj=True`).

### 3.4 `ppuu/eval_mpc.py` (Modified)
*   **Purpose**: Entry point for MPC evaluation.
*   **Modifications**:
    *   **Debugging Support**: Added `dataset_one_episode` argument to allow debugging on a single specific episode.
    *   **Integration**: Integrated with the modified `MPCKMPolicy` to support the enhanced visualization features.
