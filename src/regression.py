"""
regression.py — Linear regression to predict redshift from photometric bands.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[Regression] Linear Regression trained.")
    return model


def evaluate(model, X_test, y_test, output_dir: str):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[Regression] MSE: {mse:.4f} | R²: {r2:.4f}")

    metrics = {"MSE": round(mse, 4), "R2": round(r2, 4)}
    with open(f"{output_dir}/regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Regression] Metrics saved → {output_dir}/regression_metrics.json")

    # Predicted vs Actual scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual Redshift")
    ax.set_ylabel("Predicted Redshift")
    ax.set_title(f"Linear Regression — Redshift  (MSE={mse:.3f}, R²={r2:.3f})")
    ax.legend()
    fig.tight_layout()
    path = f"{output_dir}/regression_scatter.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Regression] Scatter plot saved → {path}")
    return metrics
