"""
Train and evaluate YOLO segmentation models for augmentation experiments.

This script runs multiple experiments (baseline / moderate_geom / strong_geom and fruit variant),
performs a validation step with fixed conf/iou, exports `best.pt` into `yolo_weights/`,
and writes a summary CSV (one row per experiment) for easy comparison.

Maintainer:
    Meruna Yugarajah <m.yugarajah@gmail.com>
    Aleksander Michalak <aleksander1.michalak@uni-a.de>
"""

import shutil
import time
import csv
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

from ultralytics import YOLO

# yolo26n.pt is needed for AMP checks and can be copied into the Ultralytics weights cache.


# ============================================================
# 1) PATHS / SETTINGS
# ============================================================
# Repository root. This script is expected to live in ``scripts/local/``.
project_root = Path(__file__).resolve().parents[2]

# DATA_YAML is set dynamically based on the --dataset argument

# Training settings
EPOCHS = 5000
IMGSZ = 640
BATCH = 16          # RTX 3080: safe. If OOM: use 4
DEVICE = 0          # GPU device index. Use -1 for CPU.
PATIENCE = 100

# ============================================================
# 2) VAL SETTINGS (fixed for all runs)
# ============================================================
def get_val_args(data_yaml: Path) -> dict:
    """Build validation arguments shared by all experiments.

    The returned settings intentionally keep confidence threshold, IoU
    threshold, image size, batch size, and dataset split identical for all
    experiments so that validation metrics remain directly comparable.

    Args:
        data_yaml: Path to the dataset configuration file.

    Returns:
        A dictionary containing the validation arguments passed to
        ``model.val()``.
    """
    return dict(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=0.25,      # fixed confidence threshold (for P/R/F1 comparison)
        iou=0.50,       # fixed IoU threshold
        split="val",
    )

# ============================================================
# 3) EXPERIMENTS (mosaic disabled)
# ============================================================
# The experiments mainly vary geometric augmentation while keeping color and
# lighting augmentations in a realistic range.
EXPERIMENTS = [
    {
        "name": "no_augmentation",
        "params": dict(),
    },
    {
        "name": "baseline",
        "params": dict(
            hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
            fliplr=0.5,
            translate=0.1,
            scale=0.1,
        ),
    },
    {
        "name": "moderate_geom",
        "params": dict(
            hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
            fliplr=0.5,
            flipud=0.1,
            translate=0.15,
            scale=0.2,
            degrees=10.0,
        ),
    },
    {
        "name": "strong_geom",
        "params": dict(
            hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
            fliplr=0.5,
            flipud=0.2,
            translate=0.2,
            scale=0.3,
            degrees=15.0,
            shear=5.0,
            perspective=0.0005,
        ),
    },
    {
        "name": "strong_geom_fruit",
        "params": dict(
            hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
            fliplr=0.5,
            flipud=0.1,      
            translate=0.2,
            scale=0.25,      
            degrees=15.0,
            perspective=0.0003,
        ),
    },
]


# ============================================================
# 4) HELPER FUNCTIONS
# ============================================================
def format_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A formatted duration string using seconds, minutes and seconds, or
        hours, minutes, and seconds depending on the input length.
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def safe_float(x: Any) -> Optional[float]:
    """Convert a value to ``float`` if possible.

    This helper is used when reading metric values from CSV files or result
    objects that may contain missing or non-numeric entries.

    Args:
        x: Value to convert.

    Returns:
        The converted float value, or ``None`` if conversion fails.
    """
    try:
        return float(x)
    except Exception:
        return None


def compute_f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    """Compute the F1 score from precision and recall.

    Args:
        p: Precision value.
        r: Recall value.

    Returns:
        The harmonic mean of precision and recall, ``0.0`` if ``p + r == 0``,
        or ``None`` if either input is missing.
    """
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def find_col(keys: List[str], candidates: List[str]) -> Optional[str]:
    """Find a matching CSV column name by case-insensitive substring search.

    Ultralytics metric column names may vary slightly between versions. This
    helper searches the available header keys and returns the first matching
    column for one of the provided candidate names.

    Args:
        keys: Available column names from the CSV header.
        candidates: Candidate metric names or substrings to search for.

    Returns:
        The first matching column name, or ``None`` if no column matches.
    """
    for cand in candidates:
        cl = cand.lower()
        for k in keys:
            if cl in k.lower():
                return k
    return None


def read_val_metrics(results_csv: Path) -> Dict[str, Optional[float]]:
    """Read validation metrics from an Ultralytics ``results.csv`` file.

    The function extracts segmentation validation metrics from a results file.
    Since Ultralytics column names can change slightly between versions,
    relevant columns are resolved through flexible substring matching.

    Extracted metrics include:
        - precision
        - recall
        - F1 score
        - mAP50
        - mAP50-95

    The function also records which columns were used to retrieve the values.

    Args:
        results_csv: Path to the ``results.csv`` file.

    Returns:
        A dictionary containing the extracted metrics and the matched column
        names. Returns an empty dictionary if the file does not exist, has no
        rows, or has no readable header.
    """
    if not results_csv.exists():
        return {}

    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows or reader.fieldnames is None:
        return {}

    keys = list(reader.fieldnames)

    # Candidate columns (masks first, then fallback)
    col_p = find_col(keys, ["metrics/precision(m)", "metrics/precision"])
    col_r = find_col(keys, ["metrics/recall(m)", "metrics/recall"])
    col_m50 = find_col(keys, ["metrics/segm_map50", "metrics/segm_mAP50", "metrics/map50"])
    col_m5095 = find_col(keys, ["metrics/segm_map50-95", "metrics/segm_mAP50-95", "metrics/map50-95"])

    def best(col: Optional[str]) -> Optional[float]:
        """Return the maximum valid numeric value from a CSV column."""
        if col is None:
            return None
        vals = []
        for r in rows:
            v = safe_float(r.get(col))
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    p = best(col_p)
    r = best(col_r)

    return {
        "precision": p,
        "recall": r,
        "f1": compute_f1(p, r),
        "mAP50": best(col_m50),
        "mAP50-95": best(col_m5095),
        "_cols_used": {"precision": col_p, "recall": col_r, "mAP50": col_m50, "mAP50-95": col_m5095},
    }


# ============================================================
# 5) SINGLE EXPERIMENT EXECUTION
# ============================================================
def run_experiment(idx: int, exp: Dict, weights_path: Path, model_name: str, dataset_name: str, run_timestamp: str, data_yaml: Path, total_experiments: int) -> Dict:
    """Train, validate, and export one experiment run.

    This function performs one complete experiment cycle:
        1. Load the base model weights.
        2. Train the model with the augmentation parameters of the experiment.
        3. Export the resulting ``best.pt`` checkpoint.
        4. Run validation with fixed settings.
        5. Extract validation metrics from the result object or CSV output.

    Args:
        idx: Zero-based index of the current experiment.
        exp: Experiment definition containing ``name`` and augmentation
            ``params``.
        weights_path: Path to the base YOLO weights file.
        model_name: Full model identifier including dataset name.
        dataset_name: Name of the selected dataset.
        run_timestamp: Shared timestamp for all runs of the current script
            execution.
        data_yaml: Path to the dataset configuration file.
        total_experiments: Total number of experiments executed in this run.

    Returns:
        A dictionary containing run metadata, output directories, exported
        checkpoint paths, extracted metrics, and run duration.

    Raises:
        FileNotFoundError: If the input weights file or generated ``best.pt``
            checkpoint cannot be found.
    """
    name = exp["name"]
    params = exp["params"]

    print("\n" + "=" * 90)
    print(f"▶ Training {idx + 1}/{total_experiments}: {name}")
    print("=" * 90)

    t0 = time.time()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    print(f"📦 Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    timestamp_underscore = run_timestamp.replace("-", "_")
    
    # Ultralytics uses `project` and `name` to build the output folder.
    # In this repo we expect segmentation runs under:
    #   runs/segment/{model_name}/{timestamp}/{experiment}/...
    # Here:
    # - `project_name` selects the model/dataset group
    # - `unique_name` encodes run timestamp (folder) and experiment name (subfolder)
    project_name = model_name  # -> runs/segment/{model_name}/
    unique_name = f"{timestamp_underscore}/{name}"

    train_args = dict(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=project_name,  
        name=unique_name,      
        amp=True,  
        plots=True,  
    )
    train_args.update(params)

    # Step 1: Train the model.
    results = model.train(**train_args)
    train_dir = Path(results.save_dir)

    # Step 2: Locate the best checkpoint generated by training.
    best_pt = train_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    # Step 3: Export a timestamped copy of the best checkpoint.
    out_dir = project_root / "yolo_weights"
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exported = out_dir / f"{model_name}-demo-dataset-{name}-{ts}.pt"
    shutil.copy2(best_pt, exported)
    print(f"✅ Exported best.pt to: {exported}")

    # Step 4: Run validation with fixed settings for a fair comparison.
    print("🔍 Validation (fixed conf/iou) ...")
    val_model = YOLO(str(best_pt))
    val_name = f"{timestamp_underscore}/VAL_{name}"
    val_args = {**get_val_args(data_yaml), "project": project_name, "name": val_name}
    val_results = val_model.val(**val_args)
    val_dir = Path(val_results.save_dir)

    # Step 5: Extract metrics from the validation result object first and fall
    # back to ``results.csv`` if necessary.
    metrics = {}
    
    if hasattr(val_results, "seg"):
        seg = val_results.seg
        p = safe_float(getattr(seg, "p", None))
        r = safe_float(getattr(seg, "r", None))
        mAP50 = safe_float(getattr(seg, "map50", None))
        mAP50_95 = safe_float(getattr(seg, "map", None))
        
        if p is not None and r is not None:
            metrics["precision"] = p
            metrics["recall"] = r
            metrics["f1"] = compute_f1(p, r)
            metrics["mAP50"] = mAP50
            metrics["mAP50-95"] = mAP50_95
    
    if not metrics.get("precision"):
        csv_metrics = read_val_metrics(val_dir / "results.csv")
        if csv_metrics.get("precision") is not None:
            metrics = csv_metrics

    dt = time.time() - t0
    print(f"⏱ Duration for run '{name}': {format_time(dt)}")

    if metrics and metrics.get("precision") is not None:
        print(
            f"📊 Val(M): P={metrics.get('precision'):.3f}  "
            f"R={metrics.get('recall'):.3f}  "
            f"F1={metrics.get('f1'):.3f}  "
            f"mAP50={metrics.get('mAP50'):.3f}  "
            f"mAP50-95={metrics.get('mAP50-95'):.3f}"
        )
    else:
        print("⚠️ Could not read metrics from results.csv (column names may differ).")

    return {
        "name": name,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "best_pt": best_pt,
        "exported": exported,
        "metrics": metrics,
        "duration_s": dt,
    }


# ============================================================
# 6) Main: run experiments sequentially + summary CSV + best model
# ============================================================
def main():
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(description='Train YOLO Segmentation Model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo11n-seg', 'yolo26n-seg'],
        default='yolo11n-seg',
        help='YOLO model choice: yolo11n-seg or yolo26n-seg (default: yolo11n-seg)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['data_420', 'data_640', 'fruit_dataset_640', 'data_640_demo_day'],
        default='data_640',
        help='Dataset choice: data_420, data_640, fruit_dataset_640, or data_640_demo_day (default: data_640)'
    )
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['no_augmentation', 'baseline', 'moderate_geom', 'strong_geom', 'strong_geom_fruit', 'all'],
        default='all',
        help='Augmentation choice: no_augmentation, baseline, moderate_geom, strong_geom, strong_geom_fruit, or all (default: all)'
    )
    args = parser.parse_args()
    
    # Filter experiments based on --augmentation
    if args.augmentation == 'all':
        experiments_to_run = EXPERIMENTS
    else:
        experiments_to_run = [exp for exp in EXPERIMENTS if exp['name'] == args.augmentation]
        if not experiments_to_run:
            raise ValueError(f"Augmentation '{args.augmentation}' not found!")
    
    dataset_name = args.dataset
    DATA_YAML = project_root / dataset_name / "data.yaml"
    
    base_model_name = args.model
    model_name = f"{base_model_name}_{dataset_name}"
    WEIGHTS_PATH = (project_root / "yolo_weights" / f"{base_model_name}.pt").resolve()
    
    # Quick sanity checks to avoid running with invalid paths
    print("=" * 90)
    print(f"🚀 YOLO training with model: {base_model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Augmentation: {args.augmentation} ({len(experiments_to_run)} Experiment(e))")
    print(f"🏷️  Model name (with dataset): {model_name}")
    print("=" * 90)
    print("project_root:", project_root)
    print("WEIGHTS_PATH:", WEIGHTS_PATH, "exists:", WEIGHTS_PATH.exists())
    print("DATA_YAML   :", DATA_YAML, "exists:", DATA_YAML.exists())
    
    # Check that the weights exist
    if not WEIGHTS_PATH.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"\n❌ Error: Weights not found: {WEIGHTS_PATH}\n\n"
            f"📥 Please download the {model_name} weights manually:\n"
            f"   URL: https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt\n"
            f"   Or: https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt\n\n"
            f"   Save the file to: {WEIGHTS_PATH}\n\n"
            f"   Example command (on a system with internet):\n"
            f"   wget https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt -O {WEIGHTS_PATH}\n"
        )
    
    print(f"✅ Weights found: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Copy `yolo26n.pt` into the Ultralytics weights cache, if needed for AMP checks.
    yolo26n_pt = project_root / "yolo_weights" / "yolo26n.pt"
    if yolo26n_pt.exists():
        try:
            from ultralytics.utils import SETTINGS
            cache_dir = Path(SETTINGS.get("weights_dir", Path.home() / ".ultralytics" / "weights"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "yolo26n.pt"
            if not cache_file.exists():
                shutil.copy2(yolo26n_pt, cache_file)
                print(f"✅ yolo26n.pt copied into the Ultralytics cache: {cache_file}")
        except Exception as e:
            print(f"⚠️ Could not copy yolo26n.pt into the cache: {e}")
    
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML}\n"
            f"Please ensure the dataset directory '{dataset_name}' exists and contains a data.yaml."
        )

    # Unique timestamp for this training run (all experiments share the same timestamp)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"📅 Run timestamp: {run_timestamp}")

    overall0 = time.time()
    all_results = []

    for i, exp in enumerate(experiments_to_run):
        all_results.append(run_experiment(i, exp, WEIGHTS_PATH, model_name, dataset_name, run_timestamp, DATA_YAML, len(experiments_to_run)))

        elapsed = time.time() - overall0
        avg = elapsed / (i + 1)
        remaining = avg * (len(experiments_to_run) - (i + 1))
        print("-" * 90)
        print(f"⏱ Total so far: {format_time(elapsed)} | ⏳ Remaining (~): {format_time(remaining)}")
        print("-" * 90)

    # Write one summary row per experiment for spreadsheet-based comparison.
    out_dir = project_root / "yolo_weights"
    out_dir.mkdir(exist_ok=True)

    summary_csv = out_dir / f"summary_{model_name}_{EPOCHS}epochs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "precision_M", "recall_M", "f1_M", "mAP50_M", "mAP50-95_M", "epochs", "weights_file"])
        for r in all_results:
            m = r.get("metrics", {})
            writer.writerow([
                r["name"],
                m.get("precision"),
                m.get("recall"),
                m.get("f1"),
                m.get("mAP50"),
                m.get("mAP50-95"),
                EPOCHS,
                str(r["exported"]),
            ])

    print(f"\n📄 Summary CSV saved: {summary_csv}")

    # Select the best run based on validation F1 score.
    best_name = None
    best_f1 = float("-inf")
    best_pt = None

    print("\n" + "#" * 90)
    print("📊 SUMMARY (only val() matters)")
    print("#" * 90)

    for r in all_results:
        m = r.get("metrics", {})
        f1v = m.get("f1")
        print(f"- {r['name']}: P={m.get('precision')} R={m.get('recall')} F1={m.get('f1')} mAP50={m.get('mAP50')}")

        if isinstance(f1v, float) and f1v > best_f1:
            best_f1 = f1v
            best_name = r["name"]
            best_pt = r["best_pt"]

    if best_pt is not None:
        stable = out_dir / f"{model_name}-demo-dataset-BEST-overall.pt"
        shutil.copy2(best_pt, stable)
        print("\n🏆 BEST (based on validation F1)")
        print(f"   Name: {best_name}")
        print(f"   F1  : {best_f1:.3f}")
        print(f"   Saved as: {stable}")
    else:
        print("\n⚠️ Could not determine the best model (F1 not available).")

    print(f"\n🎉 Done. Total runtime: {format_time(time.time() - overall0)}")


if __name__ == "__main__":
    main()
