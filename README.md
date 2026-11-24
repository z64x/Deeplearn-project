# Project 2.6: Gas Sensor Array Drift

Multi-class gas classification under temporal sensor drift using Deep Learning (`GasMLP` baseline, `Gas1DCNN` comparison, Chronological Expanding-Window CV, and UMAP dimensionality reduction).

## References

- **UCI Official Source:** [Gas Sensor Array Drift Dataset (UCI ID 224)](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
- **Kaggle Mirror [42]:** [Gas Sensor Array Drift Dataset on Kaggle](https://www.kaggle.com/datasets/qutadaalshobaki/gas-sensor-array-drift-dataset/data)
- **Starting Point Reference [27]:** [Gas Sensor Drift Detection](https://www.kaggle.com/code/dalageo/gas-sensor-drift-detection)

The dataset contains 13,910 measurements from 16 chemical sensors exposed to 6 gases (`Ethanol`, `Ethylene`, `Ammonia`, `Acetaldehyde`, `Acetone`, `Toluene`), organized chronologically into 10 batches (`batch1.dat` – `batch10.dat`) over 36 months, with 128 extracted features (`8` features × `16` sensors).

## Directory Structure

```text
.
├── data/
│   ├── raw/            # Downloaded UCI zip and batch1.dat..batch10.dat
│   └── processed/      # Parsed Parquet dataset (gas_sensor_drift.parquet)
├── src/
│   ├── data_loader.py          # Dataset download, LIBSVM parsing, Parquet caching
│   ├── models.py               # GasMLP and Gas1DCNN PyTorch architectures
│   ├── train_eval.py           # Training loop, Random 5-Fold CV, and Temporal CV
│   └── build_presentation.py   # Generates 14-slide 16:9 PowerPoint presentation
├── notebooks/
│   ├── 01_eda_gas_sensor_drift.ipynb          # Exploratory Data Analysis (Figs 1–3)
│   └── 02_experiments_mlp_cnn_umap.ipynb      # Model experiments & CV (Figs 4–7)
├── figures/            # Saved high-resolution plots (300 DPI)
├── results/            # Saved experiment tables and hyperparameters (CSV/JSON)
├── report/
│   ├── technical_report.md                        # Formal 7-section technical report
│   ├── presentation_outline.md                    # 15-minute PPT slide-by-slide outline
│   └── Gas_Sensor_Array_Drift_Presentation.pptx   # 14-slide 16:9 PowerPoint deck
├── requirements.txt
└── README.md
```

## Setup & Reproduction

1. **Create and activate a virtual environment (Windows PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. **Install PyTorch (with CUDA if available) and project dependencies:**
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

3. **Download and parse the dataset:**
   ```powershell
   python src/data_loader.py
   ```
   This downloads the UCI archive into `data/raw/`, extracts `batch1.dat`–`batch10.dat`, parses all 128 features, and saves `data/processed/gas_sensor_drift.parquet`.

4. **Run the Exploratory Data Analysis notebook:**
   ```powershell
   jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_gas_sensor_drift.ipynb
   ```

5. **Run all Deep Learning experiments (Random CV, Temporal CV, UMAP+MLP):**
   ```powershell
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/02_experiments_mlp_cnn_umap.ipynb
   ```
   This reproduces all metrics in `results/` (`random_cv_summary.csv`, `temporal_cv_per_batch.csv`, `temporal_cv_summary.csv`, `hyperparameters.json`) and Figures `04`–`07` in `figures/`.

6. **Build the PowerPoint presentation (`.pptx`):**
   ```powershell
   python src/build_presentation.py
   ```
   Generates `report/Gas_Sensor_Array_Drift_Presentation.pptx` (14 widescreen 16:9 slides with speaker notes).
