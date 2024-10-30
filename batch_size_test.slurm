#!/bin/bash -l



#SBATCH --job-name=batch_size_test_2
#SBATCH --output=batch_size_test.out
#SBATCH --error=batch_size_test.err
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu
#SBATCH --account=lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Load modules and activate conda environment
module load anaconda
module load cuda/12.1 
module load gcc

source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate llavaenv

cd /data/lhyman6/OCR/scripts_newvision/llama


# Run the script
srun python compute_optimal_batch_size.py
