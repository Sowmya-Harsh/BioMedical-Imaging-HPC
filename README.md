# Alzheimer MRI Classification with DeepSpeed

This project trains a convolutional neural network (CNN) for multi-class Alzheimer’s disease classification from MRI slices using DeepSpeed on an HPC cluster.  
The focus is on **optimizing the data loading pipeline** (multi-threaded loading, pinned memory, async prefetching) and **using mixed precision + ZeRO** for efficient training.

---

## Dataset

Source: https://www.kaggle.com/datasets/vencerlanz09/alzheimers-mri-brain-scan-images-augmented

The project uses a 4-class Alzheimer MRI image dataset with the following structure:

dataset/
├── train_images/
│ ├── mild_dementia/
│ ├── moderated_dementia/
│ ├── non_demented/
│ └── very_mild_dementia/
└── test_images/
├── MildDemented/
├── ModerateDemented/
├── NonDemented/
└── VeryMildDemented/


Each subfolder contains 2D MRI images (PNG/JPEG).  
The code maps these folders to 4 integer labels.

---

## Key Features

- **Model**: Lightweight CNN (no torchvision, custom `Dataset` and transforms)
- **Task**: 4-class MRI image classification (Alzheimer stages)
- **DeepSpeed**:
  - Mixed precision (fp16)
  - ZeRO optimization (stage 2 via `ds_config.json`)
- **HPC-aware data pipeline**:
  - `num_workers` for multi-threaded loading
  - `pin_memory=True` for faster host→GPU transfers
  - `prefetch_factor` for asynchronous prefetching
  - `non_blocking=True` on `.to(device)` calls
- **Cluster integration**:
  - Runs inside a Python virtual environment (`ds_env`)
  - Uses site-installed PyTorch + CUDA via modules
  - Launches with a SLURM batch script

---

## Project Structure

Final_Project/
├── train_alzheimer_ds.py # Main DeepSpeed training script (CNN + custom Dataset)
├── ds_config.json # DeepSpeed config (ZeRO-2 + fp16)
├── submit_deepspeed.slurm # SLURM script to launch training on the cluster
├── install_ds.sh # Optional script to create ds_env and install deepspeed
└── dataset/ # Alzheimer MRI dataset


Adjust `train_batch_size` and `train_micro_batch_size_per_gpu` according to the number of GPUs.

---
PyTorch and CUDA are always taken from the cluster modules.

## DeepSpeed Configuration (`ds_config.json`)

Example (ZeRO-2 + fp16, 2 GPUs, batch size 64 per GPU):

## SLURM Script (`submit_deepspeed.slurm`)

Example for **1 node, 2 GPUs, 12 CPUs**:

## sbatch submit_medical.slurm

## Experiments

To study the effect of the data pipeline:

- Vary `num_workers` (e.g. 0, 2, 4, 8) with fixed GPUs and batch size
- Toggle `--pin_memory` on/off
- Compare:
  - **Throughput** (images/sec per epoch)
  - **Epoch time**
  - **GPU utilization and memory usage**

All runs use the same CNN, dataset, number of GPUs, and learning rate; only data loading and memory settings change.

---

## Citation / Acknowledgements

- Alzheimer MRI dataset from Kaggle (Alzheimer’s MRI 4-class dataset).  
- DeepSpeed for distributed / mixed-precision training.  
- PyTorch for model and data pipeline.


