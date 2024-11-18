#!/bin/bash -l

#SBATCH --job-name=train_llama
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks=1               # Launch only 1 task
#SBATCH --gres=gpu:4             # Request 4 GPUs
#SBATCH --cpus-per-task=64       # Adjust CPUs per task if needed
#SBATCH --qos=qos_gpu
#SBATCH --account=lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=llama_deepspeed_%j.out
#SBATCH --error=llama_deepspeed_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

cd /data/lhyman6/OCR/scripts_newvision/llama

source ica100env/bin/activate
module load gcc/9.3.0
module load cuda/12.1

export MASTER_PORT=$(shuf -i 10000-65535 -n 1)  # Random free port
echo "MASTER_PORT=$MASTER_PORT"

mkdir -p /tmp/lhyman6/tmp/huggingface
mkdir -p /tmp/lhyman6/tmp/torch_extensions
ls /tmp/lhyman6/tmp/huggingface

# Set HuggingFace and PyTorch caches to NVMe drive
export HF_HOME=/tmp/lhyman6/tmp/huggingface
export TORCH_EXTENSIONS_DIR=/tmp/lhyman6/tmp/torch_extensions


# Set Triton cache directory
mkdir -p /tmp/lhyman6/ica100/.triton_autotune 
export TRITON_CACHE_DIR=/tmp/lhyman6/ica100/.triton_autotune
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your node's GPU IDs

# Read and export HuggingFace token
export HUGGINGFACE_TOKEN=$(cat /data/lhyman6/OCR/scripts_newvision/llama/token)

# Run the training script with DeepSpeed
deepspeed --master_port $MASTER_PORT train_llama32_deepspeed.py
