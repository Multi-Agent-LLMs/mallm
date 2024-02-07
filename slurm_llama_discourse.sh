#!/bin/sh
#SBATCH --job-name=mallm_llama_discourse
#SBATCH --output=res.log
#SBATCH --account=etechnik_gpu
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --cpus-per-gpu 2
#SBATCH --gpus 1

echo $1
echo $2

source ~/.bashrc
conda activate mallm
nvidia-smi

python3 -m torch.distributed.run --nnodes 1 --nproc_per_node 1 framework/discourse_policy/coordinator.py