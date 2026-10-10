# syntax=docker/dockerfile:1
# Cybersecurity Threat Platform: one image, two ECS services.
#   API service       -> default CMD (python api/server.py, port 5000)
#   Dashboard service -> override command in the ECS task definition:
#       ["streamlit","run","dashboard/app.py","--server.port=8501",
#        "--server.address=0.0.0.0","--server.headless=true"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create the non-root user BEFORE copying files so COPY --chown can be used.
# (A separate `chown -R` after COPY would duplicate the ~800 MB of models
# into a second image layer.)
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app \
    && chown appuser:appuser /app

WORKDIR /app

# Dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code + trained models. models/*.pkl must be the real files, not Git LFS
# pointers (~130 bytes). The Jenkinsfile runs `git lfs pull` and verifies this.
COPY --chown=appuser:appuser . .

USER appuser

# 5000 = Flask API, 8501 = Streamlit dashboard
EXPOSE 5000 8501

# Do NOT bake secrets into the image. In the ECS task definition, inject
# THREAT_PLATFORM_API_KEY from Secrets Manager / SSM, and set API_BASE_URL
# on the dashboard task to the API's address.
ENV API_BASE_URL=http://127.0.0.1:5000

# ECS ignores Dockerfile HEALTHCHECK; define health checks in the task
# definition / target group (API: GET /api/v1/health on port 5000).
CMD ["python", "api/server.py"]