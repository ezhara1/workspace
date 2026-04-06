# llama.cpp + Open WebUI setup (port 8998)

This project runs:
- `llama.cpp` server (OpenAI-compatible API) on port `8080`
- `Open WebUI` on port `8998` for browser access over the internet
- `VibeVoice` OpenAI-compatible TTS API on port `9001` (for Open WebUI voice output)

Target model repo:
`HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`

## 1) Requirements

- Linux host
- Docker + Docker Compose plugin (for Docker workflow)
- OR Python 3.10+ with `venv` support (for no-Docker workflow)
- Enough RAM/VRAM for a 35B model quantization
  - CPU-only is possible but can be very slow
  - GPU acceleration strongly recommended

## 1b) What to push to GitHub (minimal + reproducible)

Push only source/config needed to recreate the pod:
- `README.md`
- `.env.example` (not `.env`)
- `docker-compose.yml`
- `requirements.runpod.txt`
- `scripts/`

`scripts/` now includes:
- `run_manual.py` (llama.cpp + Open WebUI launcher)
- `setup_vibevoice_api.sh` (installs + starts VibeVoice OpenAI-compatible TTS API)
- `vibevoice_openai_tts_api.py` (API server used by Open WebUI TTS)

Do **not** push runtime artifacts:
- `.venv/`, `venv/`
- `models/` and `*.gguf`
- `open-webui/` runtime data
- `logs/`
- `.env`

This repo now includes a `.gitignore` that excludes those paths.

To stage only the minimal reproducible files for GitHub:

```bash
chmod +x scripts/prepare_minimal_push.sh
./scripts/prepare_minimal_push.sh
```

For VibeVoice API setup on a new Runpod, also make the script executable once:

```bash
chmod +x scripts/setup_vibevoice_api.sh
```

If large files were tracked in an earlier commit, untrack them before committing:

```bash
git rm -r --cached .venv models open-webui logs || true
git rm --cached .env .webui_secret_key || true
git add .gitignore
git commit -m "Stop tracking local runtime artifacts"
```

If the GitHub repo is already huge because old commits contain model/env files, run history cleanup (for example with `git filter-repo`) and force-push a cleaned branch.

## 2) Setup

```bash
cp .env.example .env
chmod +x scripts/setup.sh scripts/fetch_gguf.py
./scripts/setup.sh
```

What setup does:
1. Validates Docker/Compose
2. Creates `models/` and `open-webui/`
3. Downloads a `.gguf` file from the Hugging Face repo
4. Writes selected file name into `.env` as `MODEL_FILE=...`

## 2b) Setup (without Docker, recommended for Runpod pods)

```bash
cp .env.example .env
mkdir -p models open-webui logs

# Download GGUF model and set MODEL_FILE in .env
python3 scripts/fetch_gguf.py --repo "$(grep '^MODEL_REPO=' .env | cut -d'=' -f2-)" --out-dir models | tail -n1
```

### GPU runtime prep (llama.cpp Python server)

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.runpod.txt
```

## 3) Run

### Option A: Docker Compose

```bash
docker compose up -d
```

### Option B: Manual (no Docker)

Single command (recommended):

```bash
python3 scripts/run_manual.py
```

Notes:
- This command performs the full manual flow: `.env` bootstrap, model download, `.venv` + `.venv-vibevoice` setup, dependency install, and service start.
- Services are started on `:8080` (llama.cpp), `:8998` (Open WebUI), and `:9001` (VibeVoice TTS API).
- VibeVoice is started with `VIBEVOICE_REQUIRE_CUDA=true` so it uses GPU (and fails fast if CUDA is unavailable).
- For Qwen3.5 models, direct/no-thinking responses are enabled by default in manual mode (`ENABLE_THINKING=false`). In this mode `run_manual.py` launches native `llama-server` with `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --chat-template-kwargs {"enable_thinking":false}`. Set `ENABLE_THINKING=true` to restore thinking mode.
- Manual mode now uses native `llama.cpp` (`llama-server`) instead of the Python `llama_cpp.server` wrapper.
- For faster restarts after first install, use:

```bash
python3 scripts/run_manual.py --skip-install
```

Custom VibeVoice TTS port:

```bash
python3 scripts/run_manual.py --tts-port 9100
```

Load/switch model directly from a GGUF URL:

```bash
python3 scripts/run_manual.py --model-url "https://example.com/path/to/model.gguf"
```

If the URL path does not end in a filename, pass one explicitly:

```bash
python3 scripts/run_manual.py --model-url "https://example.com/download?id=abc" --model-file-name my-model.gguf
```

Manual equivalent (expanded):

```bash
set -a
source .env
set +a

# Ensure GPU offload is enabled
sed -i 's/^N_GPU_LAYERS=.*/N_GPU_LAYERS=999/' .env

# Start llama.cpp API server on :8080 (native llama-server binary)
nohup /workspace/llama.cpp/build/bin/llama-server \
  --model "/workspace/models/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size "${CTX_SIZE:-8192}" \
  --n-gpu-layers "${N_GPU_LAYERS:-999}" \
  --threads "${THREADS:-8}" \
  > logs/llama.log 2>&1 &

# Start Open WebUI on :8998
DATA_DIR=/workspace/open-webui \
WEBUI_AUTH=False \
OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=${OPENAI_API_KEY:-unused} \
DEFAULT_MODELS=${DEFAULT_MODELS:-/workspace/models/${MODEL_FILE}} \
AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST=${AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST:-120} \
nohup .venv/bin/open-webui serve --host 0.0.0.0 --port 8998 > logs/open-webui.log 2>&1 &

# Start VibeVoice OpenAI-compatible TTS API on :9001 (GPU-required)
VIBEVOICE_API_PORT=${VIBEVOICE_API_PORT:-9001} \
VIBEVOICE_REQUIRE_CUDA=true \
nohup .venv-vibevoice/bin/python scripts/vibevoice_openai_tts_api.py > logs/vibevoice-api.log 2>&1 &
```

Check status:

```bash
ss -ltnp | grep -E ':8080|:8998|:9001'
curl -sS http://127.0.0.1:8080/v1/models
curl -I http://127.0.0.1:8998
curl -sS http://127.0.0.1:9001/health
```

Open WebUI:
- `http://<your-server-ip>:8998`

llama.cpp API (direct):
- `http://<your-server-ip>:8080/v1`

VibeVoice TTS API (for Open WebUI):
- `http://<your-server-ip>:9001/v1`

VibeVoice env knobs in `.env`:
- `VIBEVOICE_API_PORT`
- `VIBEVOICE_REQUIRE_CUDA`
- `VIBEVOICE_TTS_MODEL`
- `VIBEVOICE_STRIP_THINK_FOR_TTS`

llama.cpp (manual mode) reasoning knob in `.env`:
- `ENABLE_THINKING` (`true` or `false`; defaults to `false`)

## 4) Internet exposure on port 8998

- Ensure cloud/security group/firewall allows inbound TCP `8998`
- Keep `WEBUI_AUTH=True` (default) so login is required
- For public deployments, use HTTPS via reverse proxy (Nginx/Caddy/Traefik)

## 5) Model caveat (important)

`llama.cpp` needs a **GGUF** model file.
If the Hugging Face repo has no GGUF files, setup will fail.
In that case, convert from source weights to GGUF first, then place the GGUF file in `./models` and set `MODEL_FILE` in `.env`.

## 6) Useful tuning

Edit `.env`:
- `N_GPU_LAYERS`: set >0 to offload layers to GPU
- `CTX_SIZE`: context window
- `THREADS`: CPU threads for inference
- `PARALLEL_REQUESTS`: concurrent request slots
- `ENABLE_THINKING=false`: keep direct/no-thinking responses as the manual-mode default

Then restart:

```bash
docker compose down
docker compose up -d
```

For no-Docker/manual restarts:

```bash
python3 scripts/run_manual.py --skip-install
```

## 7) VibeVoice OpenAI-compatible TTS API (for Open WebUI)

This project includes a minimal local TTS API for Open WebUI at:
- `scripts/vibevoice_openai_tts_api.py`

On a fresh Runpod, install/start it with:

```bash
chmod +x scripts/setup_vibevoice_api.sh
./scripts/setup_vibevoice_api.sh
```

Default API URL:
- `http://127.0.0.1:9001/v1`

Health check:

```bash
curl -sS http://127.0.0.1:9001/health
```

Open WebUI TTS settings:
1. Engine: `OpenAI`
2. Base URL: `http://127.0.0.1:9001/v1`
3. API Key: any non-empty value (example: `sk-local`)
4. TTS Voice: `alloy`
5. TTS Model: `microsoft/VibeVoice-Realtime-0.5B`

Think-tag control for TTS (`<think>...</think>`):
- `VIBEVOICE_STRIP_THINK_FOR_TTS=true` (default) removes think blocks before speech.
- Set `VIBEVOICE_STRIP_THINK_FOR_TTS=false` if you want TTS to read them.

Notes:
- The VibeVoice API runs in a dedicated venv: `.venv-vibevoice/`.
- It uses CUDA 12.4 PyTorch wheels and is launched with `VIBEVOICE_REQUIRE_CUDA=true`.
- Do not commit `.venv-vibevoice/`, `models/`, or `logs/`.
