FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app \
    && chown appuser:appuser /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 5000 8501
ENV API_BASE_URL=http://127.0.0.1:5000
CMD ["sh", "-c", "python scripts/seed_demo.py & exec python api/server.py"]