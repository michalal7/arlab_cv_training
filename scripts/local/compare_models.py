"""Compare trained YOLO segmentation models across augmentation experiments.

This script loads ``best.pt`` checkpoints from configured training runs,
evaluates them on the validation split with fixed parameters, and writes a CSV
comparison table containing the most relevant mask and box metrics.

The comparison is designed to be fair across experiments by using identical
validation settings for all evaluated models.

Maintainer:
    Meruna Yugarajah <m.yugarajah@gmail.com>
    Aleksander Michalak <aleksander1.michalak@uni-a.de>
"""

import csv
from pathlib import Path
from typing import Dict, Optional, List
from ultralytics import YOLO

# Validation settings are kept constant to ensure a fair comparison between
# models and augmentation variants.
project_root = Path(__file__).resolve().parents[2]
DATA_YAML = project_root / "data_640_demo_day" / "data.yaml"

# Validation parameters (kept identical for a fair comparison)
VAL_ARGS = dict(
    data=str(DATA_YAML),
    imgsz=640,
    batch=16,
    device=0,
    conf=0.25,      
    iou=0.50,       
    split="val",
    save_json=False, 
    plots=False,    
)

# Models and experiments
# IMPORTANT: model names must end with `_fruit_dataset_640` because those folders are named that way.
# MODELS = ["yolo11n-seg_fruit_dataset_640", "yolo11n-seg_data_640_demo_day", "yolo26n-seg_fruit_dataset_640"]
MODELS = ["yolo11n-seg_fruit_dataset_640", "yolo11n-seg_data_640_demo_day"]
EXPERIMENTS = ["no_augmentation", "baseline", "moderate_geom", "strong_geom", "strong_geom_fruit"]

# Timestamps per model (different for each model)
MODEL_TIMESTAMPS = {
    "yolo11n-seg_fruit_dataset_640": "20260128_190716",
    "yolo26n-seg_fruit_dataset_640": "20260128_193214",
    "yolo11n-seg_data_640_demo_day": "20260312_121409",
}


def safe_float(x: any) -> Optional[float]:
    """Convert a value to ``float`` if possible.

    This helper prevents repeated try/except blocks when reading metric values
    from CSV files or result objects.

    Args:
        x: Value to convert.

    Returns:
        The converted float value, or ``None`` if the value cannot be converted.
    """
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def compute_f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    """Compute the F1 score from precision and recall.

    The F1 score is the harmonic mean of precision and recall and is commonly
    used to summarize the trade-off between both metrics in a single value.

    Args:
        p: Precision value.
        r: Recall value.

    Returns:
        The computed F1 score, ``0.0`` if ``p + r == 0``, or ``None`` if one of
        the input values is missing.
    """
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def find_col(keys: List[str], candidates: List[str]) -> Optional[str]:
    """Find a matching metric column in a CSV header.

    Ultralytics may use slightly different metric column names across versions.
    This function performs a case-insensitive substring match and returns the
    first matching column name.

    Args:
        keys: Available column names from the CSV header.
        candidates: Candidate metric names or substrings to search for.

    Returns:
        The first matching column name, or ``None`` if no match is found.
    """
    for cand in candidates:
        cl = cand.lower()
        for k in keys:
            if cl in k.lower():
                return k
    return None


def read_val_metrics(results_csv: Path) -> Dict[str, Optional[float]]:
    """Read validation metrics from an Ultralytics ``results.csv`` file.

    The function extracts the relevant segmentation and box metrics from the
    training results file. Since column names may vary between Ultralytics
    versions, the metric columns are resolved by substring matching.

    The returned dictionary includes:
        - mask precision, recall, F1, mAP50, and mAP50-95
        - box precision, recall, mAP50, and mAP50-95

    Args:
        results_csv: Path to the ``results.csv`` file produced by training or
            validation.

    Returns:
        A dictionary containing the extracted metrics. Returns an empty
        dictionary if the file does not exist, contains no rows, or has no
        usable header information.
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
    col_p_m = find_col(keys, ["metrics/precision(m)", "metrics/precision"])
    col_r_m = find_col(keys, ["metrics/recall(m)", "metrics/recall"])
    col_m50_m = find_col(keys, ["metrics/segm_map50", "metrics/segm_mAP50", "metrics/map50"])
    col_m5095_m = find_col(keys, ["metrics/segm_map50-95", "metrics/segm_mAP50-95", "metrics/map50-95"])
    
    # Box metrics
    col_p_b = find_col(keys, ["metrics/precision(b)", "metrics/precision"])
    col_r_b = find_col(keys, ["metrics/recall(b)", "metrics/recall"])
    col_m50_b = find_col(keys, ["metrics/box_map50", "metrics/box_mAP50", "metrics/map50"])
    col_m5095_b = find_col(keys, ["metrics/box_map50-95", "metrics/box_mAP50-95", "metrics/map50-95"])

    def best(col: Optional[str]) -> Optional[float]:
        """Return the maximum valid value found in a metric column.

        Args:
            col: Name of the CSV column to evaluate.

        Returns:
            The maximum numeric value found in the column, or ``None`` if the
            column is missing or contains no valid numeric values.
        """
        if col is None:
            return None
        vals = []
        for r in rows:
            v = safe_float(r.get(col))
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    p_m = best(col_p_m)
    r_m = best(col_r_m)
    p_b = best(col_p_b)
    r_b = best(col_r_b)

    return {
        "precision_M": p_m,
        "recall_M": r_m,
        "f1_M": compute_f1(p_m, r_m),
        "mAP50_M": best(col_m50_m),
        "mAP50-95_M": best(col_m5095_m),
        "precision_B": p_b,
        "recall_B": r_b,
        "mAP50_B": best(col_m50_b),
        "mAP50-95_B": best(col_m5095_b),
    }


def evaluate_model(model_path: Path, model_name: str, experiment: str) -> Dict:
    """Evaluate a single model checkpoint and collect its metrics.

    The function first attempts to read metrics from the corresponding
    ``results.csv`` file in the training directory. If no usable metrics are
    found, it runs a fresh validation pass with the fixed comparison settings.

    Args:
        model_path: Path to the ``best.pt`` checkpoint.
        model_name: Name of the model directory under ``runs/segment/``.
        experiment: Name of the augmentation experiment.

    Returns:
        A dictionary containing model metadata and extracted metrics. Missing
        metric values are stored as ``None``. If an error occurs, the returned
        dictionary also contains an ``error`` entry with a short description.

    Raises:
        ValueError: If no timestamp is configured for the model or if the
            resolved path lies outside the expected run directory.
    """
    print(f"\n🔍 Evaluating: {model_name} - {experiment}")
    print(f"   Model: {model_path.name}")
    
    if not model_path.exists():
        print(f"   ⚠️ Model not found!")
        return {
            "model": model_name,
            "experiment": experiment,
            "precision_M": None,
            "recall_M": None,
            "f1_M": None,
            "mAP50_M": None,
            "mAP50-95_M": None,
            "precision_B": None,
            "recall_B": None,
            "mAP50_B": None,
            "mAP50-95_B": None,
        }
    
    try:
        timestamp = MODEL_TIMESTAMPS.get(model_name)
        if not timestamp:
            raise ValueError(
                f"No timestamp found for model '{model_name}'. Available: {list(MODEL_TIMESTAMPS.keys())}"
            )
        
        expected_base = project_root / "runs" / "segment" / model_name / timestamp
        if not str(model_path).startswith(str(expected_base)):
            raise ValueError(f"Model path is outside the expected runs directory: {model_path}")
        
        model = YOLO(str(model_path))
        
        
        # Expected structure: runs/segment/{model_name}/{timestamp}/{experiment}/results.csv
        # model_path is: runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt
        train_dir = model_path.parent.parent 
        
        # Validate that `train_dir` is inside the expected runs directory
        if not str(train_dir).startswith(str(expected_base)):
            raise ValueError(f"Train directory is outside the expected runs directory: {train_dir}")
        
        results_csv = train_dir / "results.csv"
        metrics = read_val_metrics(results_csv)
        
        # Fallback: if the CSV does not exist or is empty, run a new validation pass
        if not metrics.get("f1_M"):
            print("   🔄 Running a new validation pass (CSV not found or empty)...")
            val_args = VAL_ARGS.copy()
            val_args['plots'] = True
            val_args['project'] = str(project_root / "evaluation")
            val_args['name'] = f"val_{model_name}_{experiment}"
            results = model.val(**val_args, verbose=False)
            
            if hasattr(results, 'seg'):
                seg = results.seg
                p_m = safe_float(getattr(seg, 'p', None))
                r_m = safe_float(getattr(seg, 'r', None))
                map50_m = safe_float(getattr(seg, 'map50', None))
                map5095_m = safe_float(getattr(seg, 'map', None))
                
                if p_m is not None and r_m is not None:
                    metrics["precision_M"] = p_m
                    metrics["recall_M"] = r_m
                    metrics["f1_M"] = compute_f1(p_m, r_m)
                    metrics["mAP50_M"] = map50_m
                    metrics["mAP50-95_M"] = map5095_m
            
            if hasattr(results, 'box'):
                box = results.box
                metrics["precision_B"] = safe_float(getattr(box, 'p', None))
                metrics["recall_B"] = safe_float(getattr(box, 'r', None))
                metrics["mAP50_B"] = safe_float(getattr(box, 'map50', None))
                metrics["mAP50-95_B"] = safe_float(getattr(box, 'map', None))
        
        result_dict = {
            "model": model_name,
            "experiment": experiment,
            **metrics
        }
        
        if metrics.get("f1_M") is not None:
            print(f"   ✅ Precision(M): {metrics.get('precision_M', 'N/A'):.3f}")
            print(f"   ✅ Recall(M): {metrics.get('recall_M', 'N/A'):.3f}")
            print(f"   ✅ F1(M): {metrics.get('f1_M', 'N/A'):.3f}")
            print(f"   ✅ mAP50(M): {metrics.get('mAP50_M', 'N/A'):.3f}")
            print(f"   ✅ mAP50-95(M): {metrics.get('mAP50-95_M', 'N/A'):.3f}")
        else:
            print(f"   ⚠️ No metrics available")
        
        return result_dict
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": model_name,
            "experiment": experiment,
            "precision_M": None,
            "recall_M": None,
            "f1_M": None,
            "mAP50_M": None,
            "mAP50-95_M": None,
            "precision_B": None,
            "recall_B": None,
            "mAP50_B": None,
            "mAP50-95_B": None,
            "error": str(e),
        }


def main():
    print("=" * 90)
    print("📊 MODEL COMPARISON: All augmentation variants")
    print("=" * 90)
    print(f"Validation parameters: conf={VAL_ARGS['conf']}, iou={VAL_ARGS['iou']}")
    print(f"Dataset: {DATA_YAML}")
    print("Timestamps per model:")
    for model_name, timestamp in MODEL_TIMESTAMPS.items():
        print(f"  - {model_name}: {timestamp}")
    
    # Create evaluation directory for all outputs
    evaluation_dir = project_root / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {evaluation_dir}")
    
    all_results = []
    
    # Evaluate all models
    for model_name in MODELS:
        timestamp = MODEL_TIMESTAMPS.get(model_name)
        if not timestamp:
            print(f"⚠️ No timestamp found for {model_name}, skipping...")
            continue
            
        for experiment in EXPERIMENTS:
            # Find the `best.pt` checkpoint for this run/experiment
            # Expected structure: runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt
            # IMPORTANT: only use runs from the configured folders/timestamps
            model_path = (
                project_root / "runs" / "segment" / model_name / 
                timestamp / experiment / "weights" / "best.pt"
            )
            
            # Validate that the path is inside the expected runs directory
            expected_base = project_root / "runs" / "segment" / model_name / timestamp
            if not str(model_path).startswith(str(expected_base)):
                raise ValueError(f"Path is outside the expected runs directory: {model_path}")
            
            result = evaluate_model(model_path, model_name, experiment)
            all_results.append(result)
    
    
    # Use a combined timestamp for the filename
    combined_timestamp = "_".join(sorted(MODEL_TIMESTAMPS.values()))
    output_csv = evaluation_dir / f"model_comparison_{combined_timestamp}.csv"
    
    fieldnames = [
        "model", "experiment",
        "precision_M", "recall_M", "f1_M", "mAP50_M", "mAP50-95_M",
        "precision_B", "recall_B", "mAP50_B", "mAP50-95_B"
    ]
    
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            row = {k: result.get(k) for k in fieldnames}
            writer.writerow(row)
    
    print(f"\n📄 Comparison saved: {output_csv}")
    
    print("\n" + "=" * 90)
    print("📊 COMPARISON OVERVIEW")
    print("=" * 90)
    
    # Sort by F1 (Mask) for a clearer overview
    valid_results = [r for r in all_results if r.get("f1_M") is not None]
    sorted_results = sorted(valid_results, key=lambda x: x.get("f1_M", 0), reverse=True)
    
    print("\n🏆 Ranking by F1-score (Mask):")
    print("-" * 90)
    print(f"{'Experiment':<20} | {'Model':<15} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7} | {'mAP50-95':<8}")
    print("-" * 90)
    
    for r in sorted_results:
        exp = r["experiment"]
        model = r["model"]
        f1 = r.get("f1_M", 0)
        p = r.get("precision_M", 0)
        r_val = r.get("recall_M", 0)
        mAP50 = r.get("mAP50_M", 0)
        mAP50_95 = r.get("mAP50-95_M", 0)
        
        print(f"{exp:<20} | {model:<15} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f} | {mAP50_95:<8.3f}")
    
    print("\n" + "=" * 90)
    print("📈 COMPARISON BY EXPERIMENT")
    print("=" * 90)
    
    for exp in EXPERIMENTS:
        exp_results = [r for r in all_results if r["experiment"] == exp and r.get("f1_M") is not None]
        if exp_results:
            exp_results_sorted = sorted(exp_results, key=lambda x: x.get("f1_M", 0), reverse=True)
            print(f"\n🔬 {exp.upper()}:")
            print("-" * 90)
            print(f"{'Model':<15} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7} | {'mAP50-95':<8}")
            print("-" * 90)
            for r in exp_results_sorted:
                model = r["model"]
                f1 = r.get("f1_M", 0)
                p = r.get("precision_M", 0)
                r_val = r.get("recall_M", 0)
                mAP50 = r.get("mAP50_M", 0)
                mAP50_95 = r.get("mAP50-95_M", 0)
                print(f"{model:<15} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f} | {mAP50_95:<8.3f}")
        else:
            print(f"\n🔬 {exp.upper()}: ⚠️ No metrics available")
    
    print("\n" + "=" * 90)
    print("🤖 COMPARISON BY MODEL")
    print("=" * 90)
    
    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model and r.get("f1_M") is not None]
        if model_results:
            model_results_sorted = sorted(model_results, key=lambda x: x.get("f1_M", 0), reverse=True)
            print(f"\n🔬 {model.upper()}:")
            print("-" * 90)
            print(f"{'Experiment':<20} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7} | {'mAP50-95':<8}")
            print("-" * 90)
            for r in model_results_sorted:
                exp = r["experiment"]
                f1 = r.get("f1_M", 0)
                p = r.get("precision_M", 0)
                r_val = r.get("recall_M", 0)
                mAP50 = r.get("mAP50_M", 0)
                mAP50_95 = r.get("mAP50-95_M", 0)
                print(f"{exp:<20} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f} | {mAP50_95:<8.3f}")
        else:
            print(f"\n🔬 {model.upper()}: ⚠️ No metrics available")
    
    if sorted_results:
        best_row = sorted_results[0]
        print("\n" + "=" * 90)
        print("🏆 BEST OVERALL MODEL")
        print("=" * 90)
        print(f"Model: {best_row['model']}")
        print(f"Experiment: {best_row['experiment']}")
        print(f"F1-Score (Mask): {best_row['f1_M']:.3f}")
        print(f"Precision (Mask): {best_row['precision_M']:.3f}")
        print(f"Recall (Mask): {best_row['recall_M']:.3f}")
        print(f"mAP50 (Mask): {best_row['mAP50_M']:.3f}")
        print(f"mAP50-95 (Mask): {best_row['mAP50-95_M']:.3f}")
        
        best_timestamp = MODEL_TIMESTAMPS.get(best_row['model'])
        if best_timestamp:
            best_model_path = (
                project_root / "runs" / "segment" / best_row['model'] / 
                best_timestamp / best_row['experiment'] / "weights" / "best.pt"
            )
            print(f"\nBest model path: {best_model_path}")
    
    print("\n" + "=" * 90)
    print("✅ Comparison completed!")
    print("=" * 90)
    print(f"\n💡 Tips for interpretation:")
    print("   - F1-score: balance between Precision and Recall (higher = better)")
    print("   - mAP50: mean average precision at IoU=0.5 (important for detection)")
    print("   - mAP50-95: mean average precision averaged over IoU 0.5-0.95 (stricter)")
    print("   - Precision: how many predicted objects are correct?")
    print("   - Recall: how many ground-truth objects were found?")
    print("\n   For segmentation, mask metrics (M) are more important than box metrics (B)")
    print("   A high F1-score together with a high mAP50-95 is ideal!")
    print(f"\n📄 Detailed results in: {output_csv}")
    print(f"📁 All outputs saved in: {evaluation_dir}")


if __name__ == "__main__":
    main()
