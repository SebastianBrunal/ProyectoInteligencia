# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Instalar dependencias ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar código fuente ──────────────────────────────────────────────────────
COPY main.py .
COPY app.py .
COPY src/ ./src/
COPY tests/ ./tests/

# ── Copiar dataset ────────────────────────────────────────────────────────────
COPY data.csv .

# ── Crear carpeta de outputs ──────────────────────────────────────────────────
RUN mkdir -p outputs

# ── Variables de entorno ──────────────────────────────────────────────────────
ENV DATA_PATH=data.csv
ENV OUTPUT_DIR=outputs

# ── Puerto de Streamlit ───────────────────────────────────────────────────────
EXPOSE 8501

# ── Comando: correr pipeline y luego lanzar la app ────────────────────────────
CMD ["sh", "-c", "python main.py --data data.csv --output outputs && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]