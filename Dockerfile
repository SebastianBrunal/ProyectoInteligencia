# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory inside the container ───────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Create outputs directory ──────────────────────────────────────────────────
RUN mkdir -p outputs

# ── Default command: run the full pipeline ────────────────────────────────────
CMD ["python", "main.py", "--data", "data.csv", "--output", "outputs"]
