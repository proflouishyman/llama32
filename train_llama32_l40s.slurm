#!/bin/bash -l

#SBATCH --job-name=train_llama
#SBATCH --time=24:00:00
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --qos=qos_gpu
#SBATCH --account=lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=llama_deepspeed_%j.out
#SBATCH --error=llama_deepspeed_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

cd /data/lhyman6/OCR/scripts_newvision/llama


source l40senv/bin/activate #ignore the conda warnings. Use the venv.
module load gcc/9.3.0
module load cuda/12.5

# Set HuggingFace and PyTorch caches to NVMe drive
export HF_HOME=/tmp/lhyman6/tmp/huggingface
export TORCH_EXTENSIONS_DIR=/tmp/lhyman6/tmp/torch_extensions
mkdir -p /tmp/lhyman6/tmp/huggingface
mkdir -p /tmp/lhyman6/tmp/torch_extensions

# Set Triton cache directory
mkdir -p /tmp/lhyman6/ica100/.triton_autotune 
export TRITON_CACHE_DIR=/tmp/lhyman6/ica100/.triton_autotune

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your node's GPU IDs

# Set environment variables for PyTorch distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)  # Random free port
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"

# Read and export HuggingFace token
export HUGGINGFACE_TOKEN=$(cat /data/lhyman6/OCR/scripts_newvision/llama/token)

# Run Deepspeed with srun
srun deepspeed /data/lhyman6/OCR/scripts_newvision/llama/train_llama32_deepspeed.py #srun is not available? ?
