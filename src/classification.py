"""
classification.py — KNN (k=5) classifier for SDSS object classes.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_knn(X_train, y_train, k: int = 5) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"[Classification] KNN (k={k}) trained.")
    return model


def evaluate(model, X_test, y_test, label_encoder, output_dir: str):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_

    print(f"[Classification] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save metrics JSON
    metrics = {
        "accuracy": round(acc, 4),
        "classification_report": classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        ),
    }
    with open(f"{output_dir}/classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Classification] Metrics saved → {output_dir}/classification_metrics.json")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — KNN k=5  (Accuracy: {acc:.2%})")
    fig.tight_layout()
    path = f"{output_dir}/confusion_matrix.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Classification] Confusion matrix saved → {path}")
    return metrics
