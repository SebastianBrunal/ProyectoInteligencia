"""
app.py — Streamlit dashboard para el pipeline ML del dataset SDSS.
Muestra métricas, gráficas y permite predecir nuevos objetos astronómicos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_data, get_classification_data, get_regression_data, get_clustering_data
from src.classification import train_knn
from src.regression import train_linear_regression
from src.clustering import train_kmeans

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="SDSS ML Dashboard",
    page_icon="🔭",
    layout="wide"
)

# ── Cargar modelos (cacheado para no reentrenar cada vez) ─────────────────────
@st.cache_resource
def load_models():
    df = load_data("data.csv")

    X_train, X_test, y_train, y_test, le, scaler = get_classification_data(df)
    knn = train_knn(X_train, y_train, k=5)

    X_train_r, X_test_r, y_train_r, y_test_r, scaler_r = get_regression_data(df)
    lr = train_linear_regression(X_train_r, y_train_r)

    return knn, lr, le, scaler, scaler_r

knn, lr, le, scaler, scaler_r = load_models()

# ── Título ────────────────────────────────────────────────────────────────────
st.title("🔭 SDSS ML Dashboard")
st.markdown("Pipeline de Machine Learning sobre el dataset astronómico **Sloan Digital Sky Survey**.")

st.divider()

# ── Tabs principales ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Resultados del Pipeline", "🔮 Predicción", "📁 Dataset"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Resultados
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Resultados del Pipeline ML")

    col1, col2, col3 = st.columns(3)

    # Métricas de clasificación
    clf_path = "outputs/classification_metrics.json"
    if os.path.exists(clf_path):
        with open(clf_path) as f:
            clf = json.load(f)
        col1.metric("🎯 Accuracy (KNN)", f"{clf.get('accuracy', 0):.2%}")
    else:
        col1.warning("Corre el pipeline primero")

    # Métricas de regresión
    reg_path = "outputs/regression_metrics.json"
    if os.path.exists(reg_path):
        with open(reg_path) as f:
            reg = json.load(f)
        col2.metric("📈 R² Regresión", f"{reg.get('r2', 0):.4f}")
        col3.metric("📉 MSE Regresión", f"{reg.get('mse', 0):.4f}")
    else:
        col2.warning("Corre el pipeline primero")

    st.divider()

    # Gráficas
    col_a, col_b, col_c = st.columns(3)

    cm_path = "outputs/confusion_matrix.png"
    if os.path.exists(cm_path):
        col_a.subheader("Matriz de Confusión")
        col_a.image(Image.open(cm_path), use_container_width=True)

    scatter_path = "outputs/regression_scatter.png"
    if os.path.exists(scatter_path):
        col_b.subheader("Regresión — Scatter")
        col_b.image(Image.open(scatter_path), use_container_width=True)

    cluster_path = "outputs/clustering_comparison.png"
    if os.path.exists(cluster_path):
        col_c.subheader("Clustering — KMeans")
        col_c.image(Image.open(cluster_path), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Predicción
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Predecir un nuevo objeto astronómico")
    st.markdown("Ingresa los valores fotométricos del objeto para clasificarlo y estimar su redshift.")

    col1, col2 = st.columns(2)

    with col1:
        u = st.number_input("Banda u", min_value=10.0, max_value=35.0, value=19.5, step=0.01)
        g = st.number_input("Banda g", min_value=10.0, max_value=35.0, value=19.3, step=0.01)
        r = st.number_input("Banda r", min_value=10.0, max_value=35.0, value=19.1, step=0.01)

    with col2:
        i = st.number_input("Banda i", min_value=10.0, max_value=35.0, value=19.0, step=0.01)
        z = st.number_input("Banda z", min_value=10.0, max_value=35.0, value=18.9, step=0.01)
        snr = st.number_input("SNR (r)", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
        ext = st.number_input("Extinction (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.001)

    if st.button("🚀 Predecir", use_container_width=True):
        features_clf = np.array([[u, g, r, i, z, snr, ext]])
        features_reg = np.array([[u, g, r, i, z, ext]])

        # Escalar
        features_clf_scaled = scaler.transform(features_clf)
        features_reg_scaled = scaler_r.transform(features_reg)

        # Predicciones
        clase_pred = le.inverse_transform(knn.predict(features_clf_scaled))[0]
        redshift_pred = lr.predict(features_reg_scaled)[0]

        st.divider()
        res1, res2 = st.columns(2)

        emoji = {"Galaxy": "🌌", "Star": "⭐", "QSO": "💫"}.get(clase_pred, "🔭")
        res1.success(f"**Clase predicha:** {emoji} {clase_pred}")
        res2.info(f"**Redshift estimado:** `{redshift_pred:.4f}`")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📁 Vista del Dataset SDSS")

    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total filas", len(df))
        col2.metric("Columnas", len(df.columns))
        col3.metric("Clases", df["class"].nunique())

        st.markdown("**Distribución de clases:**")
        st.bar_chart(df["class"].value_counts())

        st.markdown("**Primeras filas:**")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("No se encontró data.csv")