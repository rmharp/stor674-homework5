# STOR 674 Homework 5

Containerized PyTorch CPU/GPU benchmark for Longleaf (UNC). The image is built with CUDA-enabled PyTorch and can be run locally with Docker or on Longleaf via Apptainer and Slurm.

## Quick start (local, Docker)

```bash
# build (forces linux/amd64)
docker build --platform linux/amd64 -t rmharp23/stor674-homework5:latest .

# run
docker run --rm --platform linux/amd64 rmharp23/stor674-homework5:latest
```

Or pull the prebuilt image:
```bash
docker pull rmharp23/stor674-homework5:latest
docker run --rm --platform linux/amd64 rmharp23/stor674-homework5:latest
```

## Longleaf (Apptainer + Slurm)

```bash
module load apptainer
apptainer pull --name stor674-hw5.sif docker://rmharp23/stor674-homework5:latest
```

Interactive test on a GPU node:
```bash
srun -p gpu --qos=gpu_access --gres=gpu:1 --time=00:10:00 --pty bash -l
apptainer exec --nv ~/STOR674/stor674-hw5.sif python ~/STOR674/Homework5/compute_bench.py
```

Batch job:
```bash
sbatch run_compute_bench.sh
```

## Repository contents

- `compute_bench.py` — timing benchmark (CPU and, if available, GPU)
- `Dockerfile` — CUDA-enabled PyTorch base; runs the benchmark
- `run_compute_bench.sh` — Slurm script to run on Longleaf with Apptainer
- `Homework5.Rmd` — assignment write-up with commands and outputs
- `media/` — screenshots referenced in the report

## Links

- Docker Hub image: `rmharp23/stor674-homework5:latest`
- GitHub repo: https://github.com/rmharp/stor674-homework5