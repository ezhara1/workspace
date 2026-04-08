#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VIBEVOICE_VENV_DIR:-${ROOT_DIR}/.venv-vibevoice}"
PORT="${VIBEVOICE_API_PORT:-9001}"
REQUIRE_CUDA="${VIBEVOICE_REQUIRE_CUDA:-true}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/models"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -U pip
"${VENV_DIR}/bin/pip" install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
"${VENV_DIR}/bin/pip" install "vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git" fastapi "uvicorn[standard]" python-multipart requests numpy

pkill -f "${ROOT_DIR}/scripts/vibevoice_openai_tts_api.py" || true
VIBEVOICE_API_PORT="${PORT}" VIBEVOICE_REQUIRE_CUDA="${REQUIRE_CUDA}" nohup "${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/vibevoice_openai_tts_api.py" > "${ROOT_DIR}/logs/vibevoice-api.log" 2>&1 &

sleep 2
curl -sS "http://127.0.0.1:${PORT}/health" || true

echo
echo "VibeVoice API started on: http://127.0.0.1:${PORT}/v1"
echo "VIBEVOICE_REQUIRE_CUDA=${REQUIRE_CUDA}"
DEFAULT_TTS_VOICE="${VIBEVOICE_TTS_VOICES:-en-Carter_man}"
DEFAULT_TTS_VOICE="${DEFAULT_TTS_VOICE%%,*}"
echo "OpenWebUI TTS voice: ${DEFAULT_TTS_VOICE}"
