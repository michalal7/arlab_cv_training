# SLURM usage (scripts/slurm)

This folder contains pre-configured SLURM wrapper scripts.

They typically:
1. `cd` into the repo root
2. activate `.venv`
3. start the matching Python entry point via `srun python3 ...`

## Training jobs

Run any of the following `sbatch` commands:

- YOLO11 training (default dataset `data_640`):
  ```bash
  sbatch scripts/slurm/slurm_train_yolo11.sh
  ```
- YOLO26 training (default dataset `data_640`):
  ```bash
  sbatch scripts/slurm/slurm_train_yolo26.sh
  ```
- YOLO11 on fruit dataset:
  ```bash
  sbatch scripts/slurm/slurm_train_yolo11_fruit.sh
  ```
- YOLO26 on fruit dataset:
  ```bash
  sbatch scripts/slurm/slurm_train_yolo26_fruit.sh
  ```
- Demo-day training (YOLO11N, `data_640_demo_day`, `strong_geom_fruit`):
  ```bash
  sbatch scripts/slurm/slurm_train_demo_day.sh
  ```
- Generic training wrapper (runs `scripts/local/train_models.py` with defaults from the Python script):
  ```bash
  sbatch scripts/slurm/slurm_train_yolo.sh
  ```

## Model comparison jobs

- Compare multiple trained runs (main comparison script):
  ```bash
  sbatch scripts/slurm/slurm_compare_models.sh
  ```
- Demo-day only comparison:
  ```bash
  sbatch scripts/slurm/slurm_compare_demo_day.sh
  ```

## Cluster-specific notes

The `#SBATCH` parameters in these scripts assume an A100 GPU setup (`--gres=gpu:a100:1`, partition `epyc-gpu`) and output logs into `logs/...`.

If your cluster differs, update the `#SBATCH` lines accordingly.

