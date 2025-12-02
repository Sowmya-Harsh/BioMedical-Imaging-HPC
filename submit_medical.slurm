#!/bin/sh -l
#SBATCH -J alz_ds           # job name
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=2
#SBATCH -c 12               # 12 CPUs
#SBATCH --gres=gpu:2        # 2 GPUs on this node
#SBATCH --time=0-4:00:00
#SBATCH -p gpu
#SBATCH --output=alz_ds_%j.out

module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

# Activate your venv with DeepSpeed, torchvision, etc.
source ds_env/bin/activate

export CUDA_HOME=/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/software/CUDA/12.6.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=== Node and CUDA info ==="
srun bash -c 'echo "--- $(hostname) ---";
              echo "CUDA_HOME=$CUDA_HOME";
              which nvcc || echo "nvcc not found";
              python -c "import torch; print(\"torch.cuda.is_available=\", torch.cuda.is_available(), \"n_gpus=\", torch.cuda.device_count())"'
echo "==========================="

DATA_DIR="/mnt/aiongpfs/users/sjanmahanthi/Final_Project/dataset"  # update if needed

echo "Running with:"
echo "  GPUs: 2"
echo "  CPUs: 12"
echo "  Data dir: $DATA_DIR"

# 2-GPU DeepSpeed run using srun, same style as CIFAR example
srun deepspeed train_alzheimer_ds.py \
  --data_dir "$DATA_DIR" \
  --output_dir "./outputs" \
  --num_epochs 20 \
  --batch_size 32 \
  --num_workers 8 \
  --pin_memory \
  --mixed_precision \
  --deepspeed_config ds_config.json
