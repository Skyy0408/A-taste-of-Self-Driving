#!/bin/bash

# DFM-KM MPC Visualization Script
# This script runs the evaluation with visualization enabled.

# 1. Set PYTHONPATH to include the current directory
export PYTHONPATH=.

# 2. Backup previous results to avoid confusion but keep history
if [ -d "results/mpc_eval_viz" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "Backing up previous results to results/mpc_eval_viz_$timestamp..."
    mv results/mpc_eval_viz "results/mpc_eval_viz_$timestamp"
fi

# 3. Run the evaluation script
# Parameters:
# - visualizer='dump': Enables the visualizer to save images/videos
# - cost.build_overlay=True: Generates the mask overlay (ego-view, road, lanes)
# - cost.build_cost_profile_and_traj=True: Generates the cost landscape
# - test_size_cap=2: Limits to 2 episodes for quick results (remove for full run)
# - num_processes=0: Runs in main process for stability and debugging
# - output_dir='results/mpc_eval_viz': Directory to save results

echo "Starting DFM-KM MPC Evaluation with Visualization..."
echo "This may take a few minutes depending on your hardware."

python ppuu/eval_mpc.py \
    --configs configs/eval_full_dataset.yaml configs/eval_trained_fm.yaml configs/dfm-km-mpc.yaml \
    --values visualizer='dump' \
             cost.build_overlay=True \
             cost.build_cost_profile_and_traj=True \
             test_size_cap=2 \
             num_processes=0 \
             output_dir='results/mpc_eval_viz'

# 4. Check results
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "Evaluation completed successfully!"
    echo "Results can be found in: results/mpc_eval_viz"
    echo ""
    echo "Videos: results/mpc_eval_viz/visualizer/videos/"
    echo "Images: results/mpc_eval_viz/visualizer/images/"
    echo "Stats:  results/mpc_eval_viz/evaluation_results_symbolic.json"
    echo "----------------------------------------------------------------"
else
    echo "Evaluation failed. Please check the output above for errors."
fi
