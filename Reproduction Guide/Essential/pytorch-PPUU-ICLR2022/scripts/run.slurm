#!/bin/bash
#
#SBATCH --job-name=run
#SBATCH --output=/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/logs/%j.out
#SBATCH --error=/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint="pascal|turing"
#SBATCH --exclude="reopen[1-5]"
#SBATCH --cpus-per-task=7
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=68000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail

eval "$(conda shell.bash hook)"
conda activate forked

srun $@
