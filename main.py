"""
main.py — Full ML pipeline for the SDSS astronomical dataset.

Usage:
    python main.py --data data.csv --output outputs/
"""

import argparse
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_data, get_classification_data, get_regression_data, get_clustering_data
from src.classification import train_knn, evaluate as eval_classification
from src.regression import train_linear_regression, evaluate as eval_regression
from src.clustering import train_kmeans, evaluate as eval_clustering


def parse_args():
    parser = argparse.ArgumentParser(description="SDSS ML Pipeline")
    parser.add_argument("--data", default="data.csv", help="Path to SDSS CSV file")
    parser.add_argument("--output", default="outputs", help="Directory for outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 55)
    print("  SDSS ML Pipeline — Starting")
    print("=" * 55)

    # 1. Load data
    df = load_data(args.data)

    # 2. Classification (KNN k=5)
    print("\n--- 1. CLASSIFICATION (KNN k=5) ---")
    X_train, X_test, y_train, y_test, le, _ = get_classification_data(df)
    knn = train_knn(X_train, y_train, k=5)
    eval_classification(knn, X_test, y_test, le, args.output)

    # 3. Regression (Linear Regression → redshift)
    print("\n--- 2. REGRESSION (Linear Regression) ---")
    X_train_r, X_test_r, y_train_r, y_test_r, _ = get_regression_data(df)
    lr = train_linear_regression(X_train_r, y_train_r)
    eval_regression(lr, X_test_r, y_test_r, args.output)

    # 4. Clustering (KMeans k=3)
    print("\n--- 3. CLUSTERING (KMeans k=3) ---")
    X_clust, true_labels, _ = get_clustering_data(df)
    km = train_kmeans(X_clust, n_clusters=3)
    eval_clustering(km, X_clust, true_labels, args.output)

    print("\n" + "=" * 55)
    print("  Pipeline complete. Results saved to:", args.output)
    print("=" * 55)


if __name__ == "__main__":
    main()
