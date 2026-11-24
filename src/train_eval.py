"""Training, evaluation, and cross-validation utilities for Gas Sensor Drift."""

from __future__ import annotations

import copy
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import umap

from src.models import Gas1DCNN, GasMLP


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_predict(
    model_fn: Callable[[], nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, float, int]:
    """Train a PyTorch classifier with internal validation early stopping and predict on X_test."""
    set_seed(seed)

    # Stratified 85/15 split; fall back to random split if any class has <2 samples
    _, counts = np.unique(y_train, return_counts=True)
    stratify_arg = y_train if np.all(counts >= 2) else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=seed,
        stratify=stratify_arg,
    )

    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

    # Avoid BatchNorm1d failure when a trailing mini-batch has size 1
    drop_last = len(tr_ds) > batch_size and (len(tr_ds) % batch_size == 1)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        epochs_trained = epoch
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_samples += xb.size(0)
        val_loss /= max(val_samples, 1)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds, axis=0), float(best_val_loss), epochs_trained


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Accuracy, Macro F1 (over classes present in y_true), and Weighted F1."""
    present_labels = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
        ),
    }


def run_random_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_builders: dict[str, Callable[[], nn.Module]],
    n_splits: int = 5,
    device: torch.device | None = None,
    seed: int = 42,
    **train_kwargs,
) -> pd.DataFrame:
    """Protocol A: Stratified K-Fold CV with StandardScaler fitted inside each fold."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records: list[dict] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X[tr_idx])
        X_te_scaled = scaler.transform(X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        for model_name, builder in model_builders.items():
            y_pred, best_val_loss, ep = train_and_predict(
                builder,
                X_tr_scaled,
                y_tr,
                X_te_scaled,
                device=device,
                seed=seed + fold_idx,
                **train_kwargs,
            )
            metrics = evaluate_predictions(y_te, y_pred)
            fold_records.append(
                {
                    "protocol": f"Random {n_splits}-Fold CV",
                    "model": model_name,
                    "fold": fold_idx,
                    "n_train": len(tr_idx),
                    "n_test": len(te_idx),
                    "best_val_loss": best_val_loss,
                    "epochs_trained": ep,
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(fold_records)
    summary = (
        fold_df.groupby("model", sort=False)[["accuracy", "macro_f1", "weighted_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model" if col[0] == "model" else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]
    return summary


def run_temporal_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device | None = None,
    umap_dim: int = 16,
    seed: int = 42,
    **train_kwargs,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Protocol B: Chronological Expanding-Window CV (Batches 1..t -> Batch t+1)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = (df["gas_id"].to_numpy(dtype=np.int64) - 1)
    batches = df["batch"].to_numpy(dtype=np.int64)

    per_batch_records: list[dict] = []
    pooled_preds: dict[str, list[np.ndarray]] = {"MLP": [], "1D-CNN": [], "UMAP+MLP": []}
    pooled_trues: list[np.ndarray] = []

    input_dim = len(feature_cols)

    for t in range(1, 10):
        tr_mask = batches <= t
        te_mask = batches == (t + 1)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_mask])
        X_te = scaler.transform(X_all[te_mask])
        y_tr = y_all[tr_mask]
        y_te = y_all[te_mask]
        pooled_trues.append(y_te)

        # 1. MLP (128-D)
        y_pred_mlp, val_mlp, ep_mlp = train_and_predict(
            lambda: GasMLP(input_dim=input_dim, num_classes=6),
            X_tr,
            y_tr,
            X_te,
            device=device,
            seed=seed + t,
            **train_kwargs,
        )
        m_mlp = evaluate_predictions(y_te, y_pred_mlp)
        pooled_preds["MLP"].append(y_pred_mlp)
        per_batch_records.append(
            {
                "train_batches": f"1-{t}",
                "test_batch": t + 1,
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "model": "MLP",
                "best_val_loss": val_mlp,
                "epochs_trained": ep_mlp,
                **m_mlp,
            }
        )

        # 2. 1D-CNN (8x16)
        y_pred_cnn, val_cnn, ep_cnn = train_and_predict(
            lambda: Gas1DCNN(num_sensors=16, features_per_sensor=8, num_classes=6),
            X_tr,
            y_tr,
            X_te,
            device=device,
            seed=seed + t,
            **train_kwargs,
        )
        m_cnn = evaluate_predictions(y_te, y_pred_cnn)
        pooled_preds["1D-CNN"].append(y_pred_cnn)
        per_batch_records.append(
            {
                "train_batches": f"1-{t}",
                "test_batch": t + 1,
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "model": "1D-CNN",
                "best_val_loss": val_cnn,
                "epochs_trained": ep_cnn,
                **m_cnn,
            }
        )

        # 3. UMAP + MLP (fit UMAP strictly on standardized train batches 1..t)
        reducer = umap.UMAP(
            n_components=umap_dim,
            n_neighbors=15,
            min_dist=0.1,
            random_state=seed,
        )
        X_tr_umap = reducer.fit_transform(X_tr).astype(np.float32)
        X_te_umap = reducer.transform(X_te).astype(np.float32)

        y_pred_umap, val_umap, ep_umap = train_and_predict(
            lambda: GasMLP(input_dim=umap_dim, num_classes=6),
            X_tr_umap,
            y_tr,
            X_te_umap,
            device=device,
            seed=seed + t,
            **train_kwargs,
        )
        m_umap = evaluate_predictions(y_te, y_pred_umap)
        pooled_preds["UMAP+MLP"].append(y_pred_umap)
        per_batch_records.append(
            {
                "train_batches": f"1-{t}",
                "test_batch": t + 1,
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "model": "UMAP+MLP",
                "best_val_loss": val_umap,
                "epochs_trained": ep_umap,
                **m_umap,
            }
        )

        print(
            f"[Batch 1-{t} -> {t+1}] "
            f"MLP acc={m_mlp['accuracy']:.3f} | "
            f"1D-CNN acc={m_cnn['accuracy']:.3f} | "
            f"UMAP+MLP acc={m_umap['accuracy']:.3f}"
        )

    per_batch_df = pd.DataFrame(per_batch_records)
    y_true_pooled = np.concatenate(pooled_trues, axis=0)

    pooled_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary_rows: list[dict] = []
    for model_name in ["MLP", "1D-CNN", "UMAP+MLP"]:
        y_pred_pooled = np.concatenate(pooled_preds[model_name], axis=0)
        pooled_results[model_name] = (y_true_pooled, y_pred_pooled)

        sub = per_batch_df[per_batch_df["model"] == model_name]
        pooled_metrics = evaluate_predictions(y_true_pooled, y_pred_pooled)
        summary_rows.append(
            {
                "model": model_name,
                "mean_batch_accuracy": float(sub["accuracy"].mean()),
                "std_batch_accuracy": float(sub["accuracy"].std()),
                "mean_batch_macro_f1": float(sub["macro_f1"].mean()),
                "std_batch_macro_f1": float(sub["macro_f1"].std()),
                "pooled_accuracy": pooled_metrics["accuracy"],
                "pooled_macro_f1": pooled_metrics["macro_f1"],
                "pooled_weighted_f1": pooled_metrics["weighted_f1"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return per_batch_df, pooled_results, summary_df
