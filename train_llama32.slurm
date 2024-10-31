#!/bin/bash -l

#SBATCH --job-name=train_llama
#SBATCH --time=24:00:00
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu
#SBATCH --account=lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu


echo "Job started on $(date)"
echo "Running on node $(hostname)"



cd /data/lhyman6/OCR/scripts_newvision/llama
source l40senv/bin/activate
export TRANSFORMERS_CACHE=/data/lhyman6/OCR/scripts_newvision/llama
export HF_HOME=/data/lhyman6/OCR/scripts_newvision/llama

python /data/lhyman6/OCR/scripts_newvision/llama/train_llama32.py
