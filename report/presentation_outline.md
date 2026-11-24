# 15-Minute Oral Presentation Outline — Project 2.6: Gas Sensor Array Drift

**Target Duration:** 15 minutes (~12 slides, ~1–1.5 min per slide)

---

### Slide 1 — Title Slide (0:45)
- **Title:** Multi-Class Gas Classification Under Temporal Sensor Array Drift
- **Subtitle:** Project 2.6 — Deep Learning Course Workshop
- **Author / Links:** Name, GitHub Repository & Public Notebook Link
- **Speaker Notes:** Introduce the problem briefly: classifying 6 gases from electronic-nose MOX sensors over 36 months of sensor aging.

### Slide 2 — Problem & Real-World Motivation (1:00)
- **Bullets:**
  - Electronic noses (16 MOX chemical sensors) enable low-cost real-time gas detection.
  - **Sensor Drift:** aging, poisoning, and humidity/temperature shifts alter sensor baselines over time.
  - Machine Learning challenge: covariate shift ($P_{\text{train}}(X) \neq P_{\text{test}}(X)$) degrades models deployed on future batches.
- **Speaker Notes:** Explain why static calibration fails in real-world chemical sensing and why deep representations + realistic validation matter.

### Slide 3 — Dataset Overview (1:15)
- **Bullets:**
  - **UCI Gas Sensor Array Drift Dataset:** 13,910 samples, 128 features (8 descriptors × 16 sensors), 6 gases.
  - Organized into **10 chronological batches** across 36 months; 0 missing values, 0 duplicates.
  - Extreme feature scale disparity (`[-23,660.63, +670,687.35]`) → requires Z-score scaling.
  - Batch imbalance: 161 (Batch 4) to 3,613 (Batch 7); **Toluene = 0 samples in Batches 3–5**.
- **Visual:** `figures/01_batch_class_distribution.png`
- **Speaker Notes:** Point out the missing Toluene in Batches 3–5 on the heatmap—this directly explains the Batch 6 accuracy drop later.

### Slide 4 — Visualizing Temporal Sensor Drift (1:15)
- **Bullets:**
  - Steady-state sensor responses (`DR` of Sensors 1–4) decay and shift from Batch 1 to Batch 10.
  - 2D PCA (68.6% variance) shows both gas separation and clear batch-to-batch drift trajectories.
- **Visual:** `figures/02_sensor_drift_over_batches.png` and `figures/03_pca_drift_visualization.png`
- **Speaker Notes:** Walk through how samples from Batch 1 vs. Batch 10 occupy shifted regions of feature space.

### Slide 5 — Methodology: MLP Baseline & 1D-CNN Extension (1:30)
- **Bullets:**
  - **MLP Baseline (`GasMLP`):** `128 → 256 → 128 → 64 → 6`, `BatchNorm1d`, `ReLU`, `Dropout(0.3)`.
  - **Extension 1 (`Gas1DCNN`):** Reshapes `128` features into `(8 channels × 16 sensors)` → 3 `Conv1d(k=3)` blocks (`32 → 64 → 128`) + dual Global Avg/Max Pooling.
  - **Training:** `AdamW` (`lr=1e-3`, `wd=1e-4`), `CosineAnnealingLR`, early stopping (`patience=12`).
- **Speaker Notes:** Explain why reshaping into `8 × 16` turns sensor array readings into a structured 1D sequence for convolutions.

### Slide 6 — Extensions 2 & 3: Temporal CV & UMAP (1:15)
- **Bullets:**
  - **Extension 2 — Chronological Expanding-Window CV:** Train on Batches `1..t`, test on Batch `t+1` ($t = 1..9$); `StandardScaler` fit strictly on train folds.
  - **Extension 3 — UMAP (`n_components=16`):** Non-linear manifold compression fit strictly on train folds (`UMAP+MLP`), plus 2D exploratory visualization.
- **Visual:** `figures/04_umap_2d_visualization.png`
- **Speaker Notes:** Contrast leakage-free temporal CV against naive Random K-Fold CV.

### Slide 7 — Key Finding 1: Random CV vs. Temporal CV (1:30)
- **Bullets:**
  - **Random 5-Fold CV:** MLP `99.31%` | 1D-CNN `99.35%` (optimistic upper bound due to drift leakage).
  - **Temporal Expanding CV:** MLP `82.57%` | 1D-CNN `81.92%` | UMAP+MLP `71.83%`.
  - **~17 percentage-point drop** proves random splits severely overestimate real-world performance.
- **Visual:** `figures/05_random_vs_temporal_cv.png` + Table from `results/random_cv_summary.csv` & `results/temporal_cv_summary.csv`
- **Speaker Notes:** Emphasize this as the central methodological takeaway of the project.

### Slide 8 — Key Finding 2: Per-Batch Generalization (1:30)
- **Bullets:**
  - **MLP** achieves the best overall temporal score (**82.57%** mean / **80.31%** pooled) and holds **77.25%** on Batch 10.
  - **1D-CNN** beats MLP on Batch 3 (**93.88%** vs. `79.32%`) and Batch 9 (**81.06%** vs. `76.81%`), but drops to `67.92%` on Batch 10.
  - **Batch 6 dip** (`76.65%` MLP) corresponds to Toluene reappearing after 3 absent batches.
- **Visual:** `figures/06_temporal_accuracy_per_batch.png`
- **Speaker Notes:** Explain why dense cross-sensor weights in the MLP generalize more steadily to late-stage Batch 10 drift than local 1D kernels.

### Slide 9 — Key Finding 3: Why UMAP+MLP Underperformed Under Drift (1:00)
- **Bullets:**
  - `UMAP+MLP` drops to **71.83%** mean (**67.13%** pooled; **51.67%** on Batch 10).
  - Out-of-sample `.transform()` relies on nearest-neighbor interpolation in the training manifold, which breaks down when drifted samples fall outside training support.
- **Speaker Notes:** Contrast UMAP's strength for exploratory visualization vs. its fragility as a frozen out-of-sample compressor under covariate shift.

### Slide 10 — Error Analysis: Class-Level Confusion Matrices (1:15)
- **Bullets:**
  - Robust classes under drift: **Ethylene (`95.0%`)**, **Acetone (`92.8%`)**, **Ammonia (`90.2%`)**.
  - Main confusion pair: **37.0%** of **Toluene** predicted as **Acetaldehyde** (and `21.7%` Acetaldehyde as Ethylene).
- **Visual:** `figures/07_confusion_matrices_temporal.png`
- **Speaker Notes:** Connect the Toluene–Acetaldehyde confusion directly back to Toluene's absence in Batches 3–5 and sensor aging overlap.

### Slide 11 — Software Architecture & Reproducibility (0:45)
- **Bullets:**
  - Modular codebase: `src/data_loader.py`, `src/models.py`, `src/train_eval.py`.
  - Two reproducible notebooks (`01_eda...`, `02_experiments...`), fixed `seed=42`, saved `hyperparameters.json` and CSV tables.
  - Runs end-to-end on PyTorch `2.6.0+cu124` (RTX 3050 GPU).
- **Speaker Notes:** Briefly show how anyone can reproduce all 7 figures and all tables from scratch.

### Slide 12 — Conclusion & Future Work (1:00)
- **Bullets:**
  - Chronological validation is mandatory for realistic sensor-drift evaluation.
  - Full-feature `GasMLP` outperforms `Gas1DCNN` and `UMAP+MLP` under 36-month drift.
  - **Future work:** Test-Time BatchNorm Adaptation, Domain-Adversarial Neural Networks (DANN), and recency-weighted ensembles.
- **Speaker Notes:** Wrap up concisely and invite questions.
