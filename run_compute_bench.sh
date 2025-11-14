#!/bin/bash
#SBATCH -J hw5-bench
#SBATCH -p gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH --mem=8G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module purge
module load apptainer
IMAGE="/nas/longleaf/home/rmharp/STOR674/stor674-hw5.sif"

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"

# Bind the submit dir as the working dir and run your script from there
apptainer exec --nv --bind "$SLURM_SUBMIT_DIR:/work" --pwd /work "$IMAGE" \
  python compute_bench.py

echo "Finished at: $(date)"