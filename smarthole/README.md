# Smarthole: Smart Pothole Detection and Severity Classification

An end-to-end machine learning system that learns road condition severity directly from MEMS accelerometer data using a **data-driven labeling strategy**.

---

## 1) Project Overview

### What this project solves
Road quality estimation is usually noisy when done manually. A file name like `rough` or `smooth` is not ground truth at the level of short signal segments. This project solves that by:
- extracting short windows from raw vertical acceleration (`az`),
- converting each window into physically meaningful statistical descriptors,
- discovering three natural severity groups with unsupervised learning,
- mapping those groups to `Smooth`, `Mild Pothole`, `Severe Pothole`,
- then training supervised classifiers on those discovered labels.

### Why this is academically important
It separates two tasks clearly:
1. **Label discovery** (unsupervised, data-driven),
2. **Prediction learning** (supervised, reproducible).

This design avoids subjective human labeling assumptions and makes the pipeline easier to justify in research reports.

---

## 2) Beginner-to-Pro Learning Path

### Beginner view (intuitive)
Think of each 1-second vibration snippet as a "fingerprint" of road roughness. The system first groups similar fingerprints automatically, then learns how to classify new fingerprints quickly.

### Intermediate view (ML pipeline)
1. Load and validate all raw sessions.
2. Segment each session independently into overlapping windows.
3. Extract 15 time-domain features from each window.
4. Run unsupervised clustering to discover 3 severity groups.
5. Rank groups by mean `std_val` and assign labels 0/1/2.
6. Split into train/val/test with stratification.
7. Train multiple classifiers, evaluate, compare, and persist models.

### Advanced view (engineering + integrity)
- No cross-file windows: prevents physically meaningless boundaries.
- No timestamp leakage into model inputs.
- Separate scaling contexts:
  - clustering scaler for label discovery,
  - supervised scaler fit on train split only.
- Reproducibility via fixed `random_state`.
- Class imbalance handled with `class_weight="balanced"` where applicable.

---

## 3) End-to-End Architecture

```text
Raw Sensor CSVs (8 sessions)
    |
    v
Per-file sliding windows
(window=50, stride=25)
    |
    v
Feature extraction (15 stats, az only)
    |
    v
Unsupervised label discovery
KMeans (default) / GMM / Percentile
    |
    v
Severity mapping by std_val ranking
0=Smooth, 1=Mild, 2=Severe
    |
    v
Stratified split + scaling
    |
    v
8 supervised classifiers
    |
    v
Metrics + plots + leaderboard + saved models
    |
    v
Inference (RoadReport on new CSV)
```

---

## 4) Dataset Details

All files must be in `data/raw/`.

| Filename | Rows | Notes |
|---|---:|---|
| `Dataset_Smooth.csv` | 3212 | Collection session |
| `raw_data_plain.csv` | 1157 | Collection session |
| `Database_rough.csv` | 3991 | Collection session |
| `Dataset_rough2.csv` | 1925 | Collection session |
| `Database_rough3.csv` | 1176 | Collection session |
| `Dataset_rough4.csv` | 584 | Collection session |
| `raw_data.csv` | 2223 | Collection session |
| `raw_data_pav_1_.csv` | 4756 | Collection session |
| **Total** | **19024** | |

### Schema
- `pc_time`: PC-side Unix time (used for ordering/report duration only)
- `esp_time`: ESP32 time in ms (used for ordering/debug only)
- `az`: Z-axis acceleration in g-force (**only signal used for model features**)

### Data quality handling
- UTF-8 BOM-safe loading (`encoding="utf-8-sig"`).
- Column-name trimming.
- Numeric coercion for `az`.
- NaN drop with per-file logging.
- required-file and required-column checks with clear exceptions.

---

## 5) Core Concept: Why Labels Must Be Data-Driven

### The ambiguity problem
A session named `smooth` may still contain transient shocks. A session named `rough` may contain calm subsegments. So file names cannot be trusted as per-window severity labels.

### Data-driven solution
For each window, we compute `std_val` and other vibration descriptors. Clustering finds groups in feature space. Then clusters are ordered by mean `std_val`:
- lowest mean `std_val` -> `0` Smooth,
- middle -> `1` Mild Pothole,
- highest -> `2` Severe Pothole.

This gives physically interpretable labels tied to signal variability, not metadata.

---

## 6) Signal Segmentation Strategy

### Sliding windows
- `window_size = 50` samples (about 1 second at ~50 Hz)
- `stride = 25` samples (50% overlap)

### Why per-file segmentation matters
Windows are generated **within each source file independently**. This avoids windows crossing session boundaries, which would create synthetic patterns that never happened physically.

---

## 7) Feature Engineering (15 Features)

These features are computed from each 50-sample `az` window:

| Feature | Formula / Method | Physical meaning |
|---|---|---|
| `mean` | `mean(az)` | Baseline offset around gravity |
| `std_val` | `std(az)` | Vibration intensity (primary severity cue) |
| `rms` | `sqrt(mean(az^2))` | Signal power density |
| `p2p` | `max(az)-min(az)` | Peak-to-peak swing |
| `kurtosis` | `kurtosis(..., fisher=True)` | Impulsiveness / sharp impacts |
| `skew` | `skew(az)` | Asymmetry of shocks |
| `energy` | `sum(az^2)` | Total window energy |
| `iqr` | `Q75-Q25` | Robust spread |
| `abs_mean` | `mean(abs(az))` | Average vibration magnitude |
| `median` | `median(az)` | Robust central value |
| `zcr` | sign-change rate | Oscillation tendency |
| `crest_factor` | `rms/(max(abs(az))+eps)` | Shape ratio between typical and peak |
| `variance` | `std^2` | Dispersion |
| `max_abs` | `max(abs(az))` | Maximum spike amplitude |
| `min_abs` | `min(abs(az))` | Minimum absolute amplitude |

Canonical feature order is maintained in `src/feature_engineering.py` via `FEATURE_NAMES`.

---

## 8) Label Discovery Algorithms

The project supports three methods (`config.yaml -> labeling.method`):

### A) `unsupervised_kmeans` (default)
1. Scale all windows with `StandardScaler`.
2. Run `KMeans(n_clusters=3, n_init=30, max_iter=500, random_state=42)`.
3. Rank clusters by mean `std_val`.
4. Convert cluster IDs to labels `{0,1,2}` by rank.
5. Save artifacts (`scaler`, model, mapping) to `models/kmeans_labeling_artifact.pkl`.

### B) `physics_percentile`
1. Use `std_val` directly.
2. Split by 33rd and 66th percentiles.
3. Assign tertile labels 0/1/2.

### C) `gmm`
1. Scale features.
2. Run `GaussianMixture(n_components=3, covariance_type="full", n_init=20, max_iter=500)`.
3. Rank components by mean `std_val`.
4. Assign labels 0/1/2 by rank.

---

## 9) Supervised Modeling Layer

After labels are discovered, the system trains 8 classifiers:
1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. SVM (RBF)
5. KNN
6. Decision Tree
7. Logistic Regression
8. Naive Bayes

### Data split policy
- Test split first (`test_size=0.20`, stratified)
- Validation split from remaining (`val_size=0.15` of full, stratified)
- Supervised scaler fit on train split only, then applied to val/test

### Selection strategy
- Train all models
- Rank by validation accuracy
- Run 5-fold CV on top-3 models
- Evaluate all trained models on held-out test set

---

## 10) Evaluation Metrics and Visual Diagnostics

Per model:
- Accuracy
- F1 weighted
- F1 macro
- ROC-AUC macro (OVR)

Generated diagnostics:
- Count + normalized confusion matrices
- Multi-model ROC overlays (one-vs-rest per class)
- Model comparison bar charts
- Feature-importance charts for tree-based models
- Overall and per-source label distributions
- Representative signal windows per severity class
- Final leaderboard CSV

---

## 11) Project Structure and Responsibility Map

```text
smarthole/
├── data/
│   ├── raw/                      # raw CSV sessions
│   └── processed/
│       └── labeled_windows.csv   # features + discovered labels
├── src/
│   ├── data_loader.py            # file validation, loading, cleaning
│   ├── feature_engineering.py    # windowing + 15 feature extraction
│   ├── labeling.py               # KMeans/GMM/percentile labeling
│   ├── preprocessing.py          # split + supervised scaling
│   ├── model.py                  # model factory, train, CV, save/load
│   ├── evaluate.py               # metrics + all plots + leaderboard
│   └── predict.py                # inference + RoadReport
├── models/                       # all .pkl artifacts
├── outputs/                      # plots, CSV reports, diagnostics
├── config.yaml                   # single source of truth
├── train.py                      # orchestrator: 7-step pipeline
├── main.py                       # CLI entrypoint
└── requirements.txt
```

---

## 12) Configuration Guide (`config.yaml`)

Everything is centralized in `config.yaml`:
- data paths and split ratios
- window size and stride
- labeling method and clustering settings
- feature list
- model hyperparameters
- output and model directories

To change behavior, edit config and rerun. No source code edits required for common experiments.

---

## 13) Quick Start

### 1. Environment setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Confirm dataset placement
Put all 8 required CSVs in `data/raw/` using exact expected names.

### 3. Train full pipeline
```bash
python main.py --mode train
```

### 4. Run inference on a CSV
```bash
python main.py --mode predict --input data/raw/raw_data.csv --model "Extra Trees"
```

### 5. Batch inference from `custom_data/` (new)
Place one or many CSV files in `custom_data/`, then run:

```bash
python main.py --mode predict --model "Extra Trees"
```

This processes all CSV files in `custom_data/`.

### 6. Predict only selected files from `custom_data/` (new)
```bash
python main.py --mode predict --model "Extra Trees" --files file1.csv file2.csv
```

You can also point to a different folder:

```bash
python main.py --mode predict --input-dir "path/to/my_csvs" --files a.csv b.csv --model "Extra Trees"
```

### 7. Use custom config
```bash
python main.py --mode train --config custom_config.yaml
```

---

## 14) Expected Outputs After Training

### Data artifacts
- `data/processed/labeled_windows.csv`

### Model artifacts
- `models/feature_scaler.pkl`
- `models/kmeans_labeling_artifact.pkl`
- `models/Random_Forest.pkl`
- `models/Extra_Trees.pkl`
- `models/Gradient_Boosting.pkl`
- `models/SVM_RBF.pkl`
- `models/KNN.pkl`
- `models/Decision_Tree.pkl`
- `models/Logistic_Regression.pkl`
- `models/Naive_Bayes.pkl`

### Evaluation artifacts
- `outputs/results_leaderboard.csv`
- `outputs/cross_validation_top3.csv`
- `outputs/labeling_method_comparison.csv`
- `outputs/confusion_matrix_*.png`
- `outputs/roc_curves_comparison.png`
- `outputs/model_comparison.png`
- `outputs/feature_importance_*.png`
- `outputs/label_distribution.png`
- `outputs/signal_per_class.png`
- `outputs/labeling_metadata.txt`
- `outputs/predictions/*_window_predictions.csv` (new: per-window labels + probabilities + indices/timestamps)

---

## 15) Example Results (Current Run)

### Test leaderboard
| Rank | Model | Accuracy | F1 Weighted | ROC-AUC |
|---:|---|---:|---:|---:|
| 1 | Logistic Regression | 0.9933 | 0.9934 | 1.0000 |
| 2 | Extra Trees | 0.9867 | 0.9867 | 0.9998 |
| 3 | SVM (RBF) | 0.9867 | 0.9866 | 0.9999 |
| 4 | Random Forest | 0.9800 | 0.9801 | 0.9992 |
| 5 | Gradient Boosting | 0.9733 | 0.9735 | 0.9990 |
| 6 | KNN | 0.9733 | 0.9732 | 0.9988 |
| 7 | Decision Tree | 0.9600 | 0.9600 | 0.9682 |
| 8 | Naive Bayes | 0.9133 | 0.9152 | 0.9975 |

### Top-3 cross-validation summary
| Model | CV F1 Weighted Mean | CV Std |
|---|---:|---:|
| Extra Trees | 0.9917 | 0.0078 |
| Random Forest | 0.9832 | 0.0159 |
| Gradient Boosting | 0.9669 | 0.0120 |

---

## 16) Inference Output: RoadReport

`RoadReport` prints:
- total windows analyzed,
- optional duration estimate from `pc_time`,
- percentage of Smooth/Mild/Severe,
- simple ASCII bars,
- road quality score:

`quality = 100 * (smooth_count + 0.5 * mild_count) / total_windows`

Interpretation:
- closer to 100 => smoother road profile,
- closer to 0 => severe roughness profile.

---

## 17) Reproducibility and Good ML Practices

- Fixed seeds (`numpy`, `random`, model `random_state`).
- Stratified splits to preserve label distribution.
- Config-driven parameters to avoid hardcoded experimentation.
- Robust loader checks for missing files/columns and BOM encoding.
- Model-specific persistence for reproducible deployment.

---

## 18) Common Troubleshooting

### Error: missing required raw files
Ensure all expected filenames exist exactly in `data/raw/`.

### Error: no windows generated
Input file may be too short for current `window_size`; reduce window size in `config.yaml`.

### Low performance after config edits
Check:
- changed labeling method,
- changed window parameters,
- class distribution shift in logs,
- chosen model compatibility with updated feature scale/distribution.

---

## 19) How to Extend This Project

### For students / beginners
- Add one new feature (for example, mean absolute deviation) and compare leaderboard deltas.
- Try all three labeling methods and inspect `labeling_method_comparison.csv`.

### For experienced ML engineers
- Add frequency-domain features (FFT band energy, spectral entropy, dominant frequency).
- Add nested CV or grouped temporal validation.
- Add calibration plots and uncertainty-aware decision thresholds.
- Add model registry and experiment tracking (MLflow/W&B).
- Build a streaming inference module for real-time edge deployment.

---

## 20) Limitations and Future Work

- Current feature set is time-domain only.
- Three-class assumption is fixed (not dynamically selected).
- Domain shift across vehicle type/sensor mount is not explicitly normalized.
- Overlapping windows can induce dependence among samples; future work can use group-aware evaluation protocols.

---

## 21) Citation / Research Framing

If you use this repository in a paper/report:
- clearly state that labels are **unsupervised-discovered** and **rank-mapped by vibration variability**,
- describe why filename/session names are not used as window-level truth,
- report both performance metrics and label diagnostics for transparency.

