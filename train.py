"""Train an early diabetes-risk classifier from tabular health measurements."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
MISSING_AS_ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class DiabetesRiskNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_demo_data(rows: int, seed: int) -> pd.DataFrame:
    """Make non-clinical synthetic data solely to test the project workflow."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "Pregnancies": rng.poisson(2.8, rows),
        "Glucose": rng.normal(120, 30, rows).clip(45, 250),
        "BloodPressure": rng.normal(72, 12, rows).clip(35, 125),
        "SkinThickness": rng.normal(25, 9, rows).clip(5, 70),
        "Insulin": rng.lognormal(4.5, 0.55, rows).clip(10, 500),
        "BMI": rng.normal(32, 7, rows).clip(15, 60),
        "DiabetesPedigreeFunction": rng.gamma(2.0, 0.25, rows).clip(0.05, 2.5),
        "Age": rng.normal(35, 12, rows).clip(18, 85),
    })
    score = (-8.0 + 0.045 * data.Glucose + 0.07 * data.BMI +
             0.025 * data.Age + 0.8 * data.DiabetesPedigreeFunction +
             0.1 * data.Pregnancies + rng.normal(0, 0.85, rows))
    probability = 1 / (1 + np.exp(-score))
    data[TARGET] = rng.binomial(1, probability)
    return data


def load_data(path: Path | None, seed: int) -> pd.DataFrame:
    if path is None:
        print("No CSV supplied: using synthetic demo data. Do not use this model clinically.")
        return make_demo_data(1_500, seed)
    data = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return data[FEATURES + [TARGET]].copy()


def clean_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = data.dropna(subset=[TARGET]).copy()
    y = data[TARGET].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only 0 (no diabetes) and 1 (diabetes).")
    x = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    for column in MISSING_AS_ZERO:
        x[column] = x[column].replace(0, np.nan)
    x = x.fillna(x.median(numeric_only=True))
    return x, y


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = clean_features(load_data(Path(args.data) if args.data else None, args.seed))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    train_x = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
    test_x = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
    train_y = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    test_y = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)

    model = DiabetesRiskNet(len(FEATURES))
    positive_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for _ in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(test_x)).numpy()
    predictions = (probabilities >= args.threshold).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "threshold": args.threshold,
        "test_samples": len(y_test),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "feature_names": FEATURES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": args.threshold,
    }
    torch.save(checkpoint, output_dir / "diabetes_risk_model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print(f"Saved model to {output_dir / 'diabetes_risk_model.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diabetes risk classifier.")
    parser.add_argument("--data", help="CSV using the Pima column names; omit for demo data.")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
