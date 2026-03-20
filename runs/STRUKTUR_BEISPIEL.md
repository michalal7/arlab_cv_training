# Beispiel-Struktur von `runs/segment/` nach Training

## Annahme (aktuell im Workspace)
- Outputs wurden von `scripts/local/train_models.py` erzeugt.
- Unter `runs/segment/` gibt es pro `{model_name}` einen Timestamp-Ordner.
- In jedem Timestamp-Ordner liegen je Experiment (`no_augmentation`, `baseline`, `moderate_geom`, `strong_geom`, optional `strong_geom_fruit`) ein Ordner plus `VAL_*`-Ordner für qualitative Validierungs-Visuals.

## Struktur (aktuell im Workspace):

```
runs/
└── segment/
    └── {model_name}/
        └── {timestamp}/
            ├── {experiment}/
            │   ├── weights/
            │   │   ├── best.pt
            │   │   └── last.pt
            │   ├── results.csv
            │   ├── results.png
            │   ├── confusion_matrix*.png
            │   ├── *F1_curve*.png / *PR_curve*.png / ...
            │   └── train_batch*.jpg
            └── VAL_{experiment}/
                ├── confusion_matrix*.png
                ├── val_batch*_labels.jpg
                └── val_batch*_pred.jpg
```
<!--
```
runs/
├── yolo11n-seg/                                      # Projekt-Ordner (project="yolo11n-seg")
│   └── segment/                                      # Task-Ordner (automatisch für Segmentation)
│       ├── train/                                    # Training-Ordner
│       │   ├── 20260125_143000/                     # Timestamp-Ordner
│       │   │   └── no_augmentation/                 # Experiment 1
│       │   │       ├── weights/
│       │   │       │   ├── best.pt
│       │   │       │   ├── last.pt
│       │   │       │   └── epoch*.pt
│       │   │       ├── results.csv
│       │   │       ├── results.png                  # Metriken-Verlauf
│       │   │       ├── confusion_matrix.png
│       │   │       ├── F1_curve.png
│       │   │       ├── PR_curve.png
│       │   │       ├── train_batch0.jpg             # YOLO-Plots
│       │   │       ├── train_batch1.jpg
│       │   │       ├── train_batch2.jpg
│       │   │       ├── labels.jpg                   # Label-Verteilung
│       │   │       └── args.yaml
│       │   │
│       │   ├── 20260125_143000/
│       │   │   └── baseline/                        # Experiment 2
│       │   │       └── [gleiche Struktur wie oben]
│       │   │
│       │   ├── 20260125_143000/
│       │   │   └── moderate_geom/                    # Experiment 3
│       │   │       └── [gleiche Struktur wie oben]
│       │   │
│       │   └── 20260125_143000/
│       │       └── strong_geom/                     # Experiment 4
│       │           └── [gleiche Struktur wie oben]
│       │
│       └── val/                                      # Validation-Ordner
│           └── 20260125_143000/
│               ├── VAL_no_augmentation/               # Val für Exp 1
│               │   ├── results.csv                    # Wichtig: Metriken werden hieraus gelesen
│               │   ├── confusion_matrix.png
│               │   ├── F1_curve.png
│               │   ├── PR_curve.png
│               │   ├── val_batch0_labels.jpg
│               │   ├── val_batch0_pred.jpg
│               │   └── args.yaml
│               │
│               ├── VAL_baseline/                      # Val für Exp 2
│               │   └── [gleiche Struktur wie oben]
│               │
│               ├── VAL_moderate_geom/                 # Val für Exp 3
│               │   └── [gleiche Struktur wie oben]
│               │
│               └── VAL_strong_geom/                    # Val für Exp 4
│                   └── [gleiche Struktur wie oben]
│
└── yolo26n-seg/                                      # Projekt-Ordner (project="yolo26n-seg")
    └── segment/                                      # Task-Ordner (automatisch für Segmentation)
        ├── train/
        │   └── 20260125_150000/                      # Timestamp-Ordner
        │       ├── no_augmentation/
        │       ├── baseline/
        │       ├── moderate_geom/
        │       └── strong_geom/
        │
        └── val/
            └── 20260125_150000/
                ├── VAL_no_augmentation/
                ├── VAL_baseline/
                ├── VAL_moderate_geom/
                └── VAL_strong_geom/

```
-->

## Kurze Zusammenfassung (für neue Nutzer)

- `{timestamp}/` gruppiert alle Experimente eines Trainingsruns.
- `{experiment}/` enthält Training-Ausgaben inkl. `weights/` und `results.csv` (dort liegen die numerischen Metriken).
- `VAL_{experiment}/` enthält v.a. Visuals (z.B. `val_batch*_labels.jpg` und `val_batch*_pred.jpg`). In deinem Workspace liegt dort typischerweise keine `results.csv`.

## Wo finde ich die Metriken für den Vergleich?

`compare_models.py` / `compare_models_demo_day.py` liest aus:
- `runs/segment/{model_name}/{timestamp}/{experiment}/results.csv`

## Beispielpfade

- best weights: `runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt`
- Val-Visuals: `runs/segment/{model_name}/{timestamp}/VAL_{experiment}/val_batch*_pred.jpg`
