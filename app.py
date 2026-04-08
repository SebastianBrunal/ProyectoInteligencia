"""
app.py — Streamlit dashboard para el pipeline ML del dataset SDSS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_data, get_classification_data, get_regression_data, get_clustering_data
from src.classification import train_knn
from src.regression import train_linear_regression
from src.clustering import train_kmeans

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SDSS ML Dashboard", page_icon="🔭", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4466;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #7eb8f7; }
    .metric-label { font-size: 0.9rem; color: #a0aec0; margin-top: 4px; }
    .section-title { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelos ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    df = load_data("data.csv")
    # Clasificacion: features = [u, g, r, i, z, redshift]
    X_train, X_test, y_train, y_test, le, scaler_clf = get_classification_data(df)
    knn = train_knn(X_train, y_train, k=5)
    # Regresion: features = [u, g, r, i, z]
    X_train_r, X_test_r, y_train_r, y_test_r, scaler_reg = get_regression_data(df)
    lr = train_linear_regression(X_train_r, y_train_r)
    return knn, lr, le, scaler_clf, scaler_reg, df

knn, lr, le, scaler_clf, scaler_reg, df = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔭 SDSS ML Dashboard")
st.markdown("Pipeline de Machine Learning sobre el dataset astronómico **Sloan Digital Sky Survey**.")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Resultados del Pipeline", "🔮 Predicción", "📁 Dataset"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Resultados
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    clf_path = "outputs/classification_metrics.json"
    reg_path = "outputs/regression_metrics.json"
    clu_path = "outputs/clustering_metrics.json"

    accuracy, r2, mse, silhouette = "—", "—", "—", "—"
    clf_data = {}

    if os.path.exists(clf_path):
        with open(clf_path) as f:
            clf_data = json.load(f)
        accuracy = f"{clf_data.get('accuracy', 0):.2%}"

    if os.path.exists(reg_path):
        with open(reg_path) as f:
            reg = json.load(f)
        r2  = f"{reg.get('R2', reg.get('r2', 0)):.4f}"
        mse = f"{reg.get('MSE', reg.get('mse', 0)):.4f}"

    if os.path.exists(clu_path):
        with open(clu_path) as f:
            clu = json.load(f)
        silhouette = f"{clu.get('silhouette', clu.get('inertia', 0)):.4f}"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy}</div><div class="metric-label">🎯 Accuracy KNN</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{r2}</div><div class="metric-label">📈 R² Regresión</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{mse}</div><div class="metric-label">📉 MSE Regresión</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{silhouette}</div><div class="metric-label">🔵 Inertia Clustering</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Confusion Matrix
    with col_a:
        st.markdown('<p class="section-title">Matriz de Confusión — KNN (k=5)</p>', unsafe_allow_html=True)
        cm_list = clf_data.get("confusion_matrix")
        labels  = clf_data.get("labels", ["Galaxy", "QSO", "Star"])
        if cm_list:
            cm = np.array(cm_list)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels,
                        linewidths=0.5, linecolor='#3d4466',
                        annot_kws={"size": 14, "weight": "bold"}, ax=ax)
            ax.set_xlabel("Predicho", color='#a0aec0')
            ax.set_ylabel("Real", color='#a0aec0')
            ax.tick_params(colors='#a0aec0')
            plt.tight_layout()
            st.pyplot(fig)
        elif os.path.exists("outputs/confusion_matrix.png"):
            st.image(Image.open("outputs/confusion_matrix.png"), use_column_width=True)

    # Distribución de clases
    with col_b:
        st.markdown('<p class="section-title">Distribución de Clases</p>', unsafe_allow_html=True)
        counts = df["class"].value_counts()
        colors = ["#7eb8f7", "#f7a278", "#78f7c2"]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#1e2130')
        ax2.set_facecolor('#1e2130')
        bars = ax2.bar(counts.index, counts.values, color=colors, edgecolor='#3d4466', linewidth=1.2, width=0.5)
        for bar, val in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha='center', color='white', fontsize=12, fontweight='bold')
        ax2.set_ylabel("Cantidad", color='#a0aec0')
        ax2.tick_params(colors='#a0aec0')
        ax2.spines[['top','right','left','bottom']].set_color('#3d4466')
        ax2.set_ylim(0, counts.max() * 1.15)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-title">Regresión Lineal — Redshift</p>', unsafe_allow_html=True)
        if os.path.exists("outputs/regression_scatter.png"):
            st.image(Image.open("outputs/regression_scatter.png"), use_column_width=True)

    with col_d:
        st.markdown('<p class="section-title">Clustering KMeans (k=3)</p>', unsafe_allow_html=True)
        X_clust, true_labels, _ = get_clustering_data(df)
        km = train_kmeans(X_clust, n_clusters=3)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor('#1e2130')
        ax3.set_facecolor('#1e2130')
        palette = ["#7eb8f7", "#f7a278", "#78f7c2"]
        for idx, c in enumerate(np.unique(km.labels_)):
            mask = km.labels_ == c
            ax3.scatter(X_clust[mask, 0], X_clust[mask, 1],
                        c=palette[idx], label=f"Cluster {c}", alpha=0.7, s=20, edgecolors='none')
        ax3.set_xlabel("u - g", color='#a0aec0')
        ax3.set_ylabel("g - r", color='#a0aec0')
        ax3.tick_params(colors='#a0aec0')
        ax3.spines[['top','right','left','bottom']].set_color('#3d4466')
        ax3.legend(facecolor='#2d3250', labelcolor='white', edgecolor='#3d4466')
        plt.tight_layout()
        st.pyplot(fig3)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Predicción
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔮 Predecir un nuevo objeto astronómico")
    st.markdown("""
    Ingresa las **5 bandas fotométricas** (u, g, r, i, z) y el **redshift** para clasificar el objeto.
    Las bandas solas estiman el redshift con regresión lineal.
    """)

    col1, col2 = st.columns(2)
    with col1:
        u = st.number_input("Banda u", min_value=10.0, max_value=35.0, value=19.5, step=0.01)
        g = st.number_input("Banda g", min_value=10.0, max_value=35.0, value=19.3, step=0.01)
        r = st.number_input("Banda r", min_value=10.0, max_value=35.0, value=19.1, step=0.01)
    with col2:
        i  = st.number_input("Banda i", min_value=10.0, max_value=35.0, value=19.0, step=0.01)
        z  = st.number_input("Banda z", min_value=10.0, max_value=35.0, value=18.9, step=0.01)
        rs = st.number_input("Redshift (para clasificación)", min_value=0.0, max_value=10.0, value=0.5, step=0.001)

    if st.button("🚀 Predecir"):
        # Clasificacion usa [u, g, r, i, z, redshift] — 6 features
        features_clf = scaler_clf.transform(np.array([[u, g, r, i, z, rs]]))
        # Regresion usa [u, g, r, i, z] — 5 features
        features_reg = scaler_reg.transform(np.array([[u, g, r, i, z]]))

        clase_pred    = le.inverse_transform(knn.predict(features_clf))[0]
        redshift_pred = lr.predict(features_reg)[0]

        st.divider()
        res1, res2 = st.columns(2)
        emoji = {"Galaxy": "🌌", "Star": "⭐", "QSO": "💫"}.get(clase_pred, "🔭")
        res1.success(f"**Clase predicha:** {emoji} {clase_pred}")
        res2.info(f"**Redshift estimado (regresión):** `{redshift_pred:.4f}`")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📁 Vista del Dataset SDSS")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total filas", len(df))
    c2.metric("Columnas", len(df.columns))
    c3.metric("Clases", df["class"].nunique())
    st.markdown("**Primeras 20 filas:**")
    st.dataframe(df.head(20), use_container_width=True)