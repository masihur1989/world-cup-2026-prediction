# Runtime image for the Streamlit dashboard (read-only; no model code).
# Works on Render, Fly.io, and Cloud Run. Streamlit Cloud ignores this file.
FROM python:3.12-slim

WORKDIR /app

# Install only the lightweight dashboard runtime deps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy just what the dashboard needs: the app, the pandas-only src module it
# imports, the fixtures, and the committed prediction snapshots.
COPY app/ ./app/
COPY src/__init__.py src/fixtures_bracket.py ./src/
COPY data/raw/wc2026_fixtures.csv ./data/raw/
COPY data/processed/predictions/ ./data/processed/predictions/

# Cloud Run / Render / Fly provide $PORT; default to 8501 locally.
ENV PORT=8501
EXPOSE 8501

# Bind to 0.0.0.0 and the platform port; headless for server environments.
CMD streamlit run app/dashboard.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
