#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=run
#SBATCH --output=/home/us441/scratch/logs/%j.out
#SBATCH --error=/home/us441/scratch/logs/%j.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --constraint="pascal|turing"
#SBATCH --ntasks-per-node=2
#SBATCH --mem=100000
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --signal=SIGUSR1@90

# activate conda env
eval "$(conda shell.bash hook)"
conda activate forked

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
#
#module purge
#module load cuda/10.1.105
#module load cudnn/10.1v7.6.5.32

srun $@
