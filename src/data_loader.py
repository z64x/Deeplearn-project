"""Download, parse, and load the UCI Gas Sensor Array Drift dataset."""

from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

GAS_LABELS: dict[int, str] = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

UCI_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
NUM_BATCHES = 10
NUM_FEATURES = 128


def download_dataset(raw_dir: str | Path = "data/raw") -> Path:
    """Download and extract batch1.dat..batch10.dat from UCI into raw_dir."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    expected_files = [raw_path / f"batch{i}.dat" for i in range(1, NUM_BATCHES + 1)]
    if all(f.exists() for f in expected_files):
        return raw_path

    zip_path = raw_path / "gas_sensor_array_drift_dataset.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from {UCI_URL} ...")
        urllib.request.urlretrieve(UCI_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            if filename.startswith("batch") and filename.endswith(".dat"):
                target = raw_path / filename
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    missing = [f.name for f in expected_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected batch files after extraction: {missing}")

    return raw_path


def parse_batch_file(path: str | Path, batch_id: int) -> pd.DataFrame:
    """Parse a single LIBSVM-formatted batch*.dat file into a DataFrame."""
    rows: list[list[float]] = []
    gas_ids: list[int] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            # Handle optional `<label>;<concentration>` prefix if present
            label_token = parts[0].split(";")[0]
            gas_id = int(float(label_token))

            features = np.zeros(NUM_FEATURES, dtype=np.float64)
            for token in parts[1:]:
                idx_str, val_str = token.split(":", maxsplit=1)
                idx = int(idx_str) - 1
                if not (0 <= idx < NUM_FEATURES):
                    raise ValueError(
                        f"Feature index {idx + 1} out of bounds in {path}:{line_num}"
                    )
                features[idx] = float(val_str)

            gas_ids.append(gas_id)
            rows.append(features.tolist())

    feature_cols = [f"f{i}" for i in range(1, NUM_FEATURES + 1)]
    df = pd.DataFrame(rows, columns=feature_cols, dtype=np.float64)
    df.insert(0, "gas_name", [GAS_LABELS[g] for g in gas_ids])
    df.insert(0, "gas_id", gas_ids)
    df.insert(0, "batch", batch_id)
    df["batch"] = df["batch"].astype(np.int64)
    df["gas_id"] = df["gas_id"].astype(np.int64)
    return df


def load_dataset(
    raw_dir: str | Path = "data/raw",
    processed_path: str | Path | None = "data/processed/gas_sensor_drift.parquet",
) -> pd.DataFrame:
    """Load all 10 batches into a single DataFrame and optionally cache to Parquet."""
    if processed_path is not None:
        cache_path = Path(processed_path)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    raw_path = download_dataset(raw_dir)
    batch_frames = [
        parse_batch_file(raw_path / f"batch{i}.dat", batch_id=i)
        for i in range(1, NUM_BATCHES + 1)
    ]
    df = pd.concat(batch_frames, ignore_index=True)

    if processed_path is not None:
        cache_path = Path(processed_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir_path = project_root / "data" / "raw"
    parquet_path = project_root / "data" / "processed" / "gas_sensor_drift.parquet"

    dataset = load_dataset(raw_dir=raw_dir_path, processed_path=parquet_path)
    feature_cols = [f"f{i}" for i in range(1, NUM_FEATURES + 1)]
    feat_values = dataset[feature_cols].to_numpy()

    print(f"Saved processed dataset to: {parquet_path}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"NaN count: {int(dataset.isna().sum().sum())}")
    print(f"Inf count: {int(np.isinf(feat_values).sum())}")
    print(f"Feature range: [{feat_values.min():.4f}, {feat_values.max():.4f}]")
    print("\nSamples per batch:")
    print(dataset["batch"].value_counts().sort_index().to_string())
    print("\nSamples per gas class:")
    print(dataset["gas_name"].value_counts().to_string())
