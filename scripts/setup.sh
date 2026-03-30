#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
ENV_EXAMPLE="${ROOT_DIR}/.env.example"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not installed."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is required but not available."
  exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  echo "Created ${ENV_FILE} from .env.example"
fi

mkdir -p "${ROOT_DIR}/models" "${ROOT_DIR}/open-webui"

MODEL_REPO="$(grep '^MODEL_REPO=' "${ENV_FILE}" | cut -d'=' -f2-)"
MODEL_FILE="$(grep '^MODEL_FILE=' "${ENV_FILE}" | cut -d'=' -f2-)"
HF_TOKEN="${HF_TOKEN:-$(grep '^HF_TOKEN=' "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2- || true)}"

TOKEN_ARGS=()
if [ -n "${HF_TOKEN}" ]; then
  TOKEN_ARGS=(--token "${HF_TOKEN}")
fi

if [ -z "${MODEL_FILE}" ]; then
  echo "MODEL_FILE is empty. Auto-discovering and downloading a GGUF model from ${MODEL_REPO}..."
  DISCOVERED_FILE="$(python3 "${ROOT_DIR}/scripts/fetch_gguf.py" --repo "${MODEL_REPO}" --out-dir "${ROOT_DIR}/models" "${TOKEN_ARGS[@]}" | tail -n1)"

  if [ -z "${DISCOVERED_FILE}" ]; then
    echo "Failed to discover/download a GGUF model file."
    exit 1
  fi

  sed -i "s|^MODEL_FILE=.*$|MODEL_FILE=${DISCOVERED_FILE}|" "${ENV_FILE}"
  echo "Set MODEL_FILE=${DISCOVERED_FILE} in .env"
else
  if [ ! -f "${ROOT_DIR}/models/${MODEL_FILE}" ]; then
    echo "MODEL_FILE is set to ${MODEL_FILE}, but file is missing in ./models. Downloading..."
    python3 "${ROOT_DIR}/scripts/fetch_gguf.py" --repo "${MODEL_REPO}" --out-dir "${ROOT_DIR}/models" --file "${MODEL_FILE}" "${TOKEN_ARGS[@]}" >/dev/null
  fi
fi

echo "Setup complete. Start services with:"
echo "  docker compose up -d"
