# =============================================================================
# Stage 1: Build the React 19 + TypeScript frontend dashboard
# =============================================================================
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json ./
RUN npm install --frozen-lockfile || npm install

COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Production Python runtime for Neural Process Twin API
# =============================================================================
FROM python:3.12-slim-bookworm AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    NPT_MAX_CONCURRENT_NUMERICAL_SOLVES=4 \
    NPT_MAX_CONCURRENT_NEURAL_INFERENCES=8 \
    NPT_REQUEST_EXECUTION_TIMEOUT_SECONDS=30.0

WORKDIR /app

# Install security updates and curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Install Python package dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install .

# Copy backend source code and scientific artifacts
COPY src/ /app/src/
COPY artifacts/checkpoints/ /app/artifacts/checkpoints/
COPY artifacts/calibration/ /app/artifacts/calibration/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Set permissions
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Healthcheck testing readiness probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/ready || exit 1

ENTRYPOINT ["uvicorn", "neural_process_twin.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
