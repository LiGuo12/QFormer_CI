#!/bin/bash
#SBATCH --job-name=iu_con
#SBATCH --account=st-zjanew-1-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4         # Request all 4 GPUs available on the node, it is V100 16GB or 32GB
#SBATCH --ntasks=4                # Launch 4 tasks, one for each GPU
#SBATCH --constraint=gpu_mem_32
#SBATCH --cpus-per-task=6         # Request all 24 CPU cores available on the node
#SBATCH --mem=180G                 # Request all available 192 GB of RAM
#SBATCH --time=20:00:00          # max time is one week, 168 hours
#SBATCH --output=iu_stage1_output.txt
#SBATCH --error=iu_stage1_error.txt
#SBATCH --mail-user=lguo@ece.ubc.ca
#SBATCH --mail-type=ALL

# change directory
cd /scratch/st-zjanew-1/li/codes/QFormer_CI
# Load virtualenv
# module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
# Activate virtualenv
source ~/.bashrc
conda activate python_39
# export TORCH_HOME=/scratch/st-zjanew-1/li/torch_cache
export TORCH_HOME=/arc/project/st-zjanew-1/li/cache/torch_cache/
export HF_HOME=/scratch/st-zjanew-1/li/huggingface
export JAVA_HOME=~/java/jdk-11
export PATH=$JAVA_HOME/bin:$PATH

# Set the number of OpenMP threads based on the allocated CPUs per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#export PYTHONUNBUFFERED=1
# export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export TORCH_NCCL_BLOCKING_WAIT=1 #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
# export MASTER_ADDR=$(hostname)
# export MASTER_PORT=34567
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo 'Start Training'
torchrun --nproc_per_node=4 train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_ci.yaml
# torchrun --nproc_per_node=2 train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_ci.yaml
# torchrun --nproc_per_node=4 train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_li.yaml
# torchrun --nproc_per_node=2 train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_li.yaml
# python train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_li.yaml
# python train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_llm.yaml
# python train_li.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_li.yaml
# salloc  --account=st-zjanew-1-gpu --partition=interactive_gpu --time=2:0:0 -N 1 -n 2 --mem=32G --gpus=2
# vim lavis/projects/blip2/train/pretrain_stage1_li.yaml



