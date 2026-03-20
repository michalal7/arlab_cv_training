# Local usage (scripts/local)

This folder contains the Python entry points that can be run directly on a workstation/login node (no SLURM wrapper).

## 1) `train_models.py` (training + validation + export)

Trains Ultralytics YOLO segmentation models for a chosen dataset and augmentation setup, runs a validation step, and exports the best checkpoint into `yolo_weights/`.

### Prerequisites

1. Create & activate a virtual environment in the repo root:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure pretrained weights exist locally:
   - `yolo_weights/yolo11n-seg.pt` for `--model yolo11n-seg`
   - `yolo_weights/yolo26n-seg.pt` for `--model yolo26n-seg`

If a weight file is missing, the script raises an error and prints download instructions.

### Command line

```bash
python3 scripts/local/train_models.py \
  --model yolo11n-seg \
  --dataset data_640 \
  --augmentation all
```

Supported options:

- `--model`: `yolo11n-seg` or `yolo26n-seg`
- `--dataset`: `data_420`, `data_640`, `fruit_dataset_640`, `data_640_demo_day`
- `--augmentation`:
  - `no_augmentation`
  - `baseline`
  - `moderate_geom`
  - `strong_geom`
  - `strong_geom_fruit`
  - `all` (runs all experiments defined in the script)

### Expected outputs

- Training runs:
  - `runs/segment/{model_name}/{timestamp}/{experiment}/...`
  - The numerics for comparisons come from:
    - `runs/segment/{model_name}/{timestamp}/{experiment}/results.csv`
- Exported best weights:
  - `yolo_weights/{model_name}-demo-dataset-<experiment-name>-<timestamp>.pt`

### Important note (runtime)

`train_models.py` uses a fixed `EPOCHS = 5000` inside the script (there is no CLI flag for epochs). Training can therefore be expensive.

## 2) `compare_models.py` (multi-model, augmentation comparison)

Evaluates multiple trained runs (expects the `runs/segment/...` layout produced by `train_models.py`) and writes a comparison table into `evaluation/`.

### Command

```bash
python3 scripts/local/compare_models.py
```

### Assumptions / expectations

- The script uses hard-coded `MODEL_TIMESTAMPS` inside the file.
- It expects checkpoints at:
  - `runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt`
- If you trained new runs after the timestamps recorded in `MODEL_TIMESTAMPS`, you may need to update that mapping in the script.

## 3) `compare_models_demo_day.py` (demo-day only comparison)

Same idea as `compare_models.py`, but focused on the `data_640_demo_day` dataset.

### Command

```bash
python3 scripts/local/compare_models_demo_day.py
```

## 4) Quick smoke test (no training)

This only checks that Python files parse correctly:

```bash
python3 -m py_compile scripts/local/*.py
```

