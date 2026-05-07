#!/usr/bin/env bash
set -euo pipefail

# --- Config (override via env) ---
IMAGE_NAME="${IMAGE_NAME:-flask-app}"
CONTAINER_NAME="${CONTAINER_NAME:-flask-app}"
PORT="${PORT:-3333}"
APP_MODULE="${APP_MODULE:-app:app}"

IMAGE_NAME="${IMAGE_NAME}_at_${PORT}"
CONTAINER_NAME="${CONTAINER_NAME}_at_${PORT}"

# --- Checks ---
if [[ ! -f "requirements.txt" ]]; then
  echo "ERROR: requirements.txt not found in $(pwd)" >&2
  exit 1
fi

# --- Write .dockerignore (NOT used by -v mount) ---
cat > .dockerignore <<'EOF'
.git
__pycache__/
*.pyc
*.pyo
*.pytest_cache
*.env
.env*
.vscode
.idea
.DS_Store
*.md
EOF

# --- Write Dockerfile ---
cat > Dockerfile <<'EOF'
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# system deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# create new user
RUN groupadd --system app \
 && useradd --system --gid app --home-dir /home/appuser --create-home \
      --shell /usr/sbin/nologin \
      --comment "App runtime user" appuser \
 && passwd -l appuser

USER appuser

# python deps
COPY --chown=appuser:app requirements.txt /app/requirements.txt
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN python -m pip install --user --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# copy app data
# COPY --chown=appuser:app . /app

ENTRYPOINT ["bash", "./gunicorn_serve.sh"]
EOF

# --- Build image ---
echo "Building image: ${IMAGE_NAME}"
podman build -t "${IMAGE_NAME}" .

# --- Stop/remove any old container with the same name ---
if podman ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
  echo "Removing existing container: ${CONTAINER_NAME}"
  podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "Running container: ${CONTAINER_NAME}"
echo "  172.17.0.1:${PORT} -> ${PORT}"
podman run --name "${CONTAINER_NAME}" --rm -it \
  --read-only \
  -v .:/app:ro \
  -p "172.17.0.1:${PORT}:${PORT}" \
  -e BIND="0.0.0.0:${PORT}" \
  "${IMAGE_NAME}"
