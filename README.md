# llama.cpp + Open WebUI setup (port 8998)

This project runs:
- `llama.cpp` server (OpenAI-compatible API) on port `8080`
- `Open WebUI` on port `8998` for browser access over the internet

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
- This command performs the full manual flow: `.env` bootstrap, model download, `.venv` setup, dependency install, and service start.
- Services are started on `:8080` (llama.cpp) and `:8998` (Open WebUI).
- For faster restarts after first install, use:

```bash
python3 scripts/run_manual.py --skip-install
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

# Start llama.cpp API server on :8080
nohup .venv/bin/python -m llama_cpp.server \
  --model "/workspace/models/${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port 8080 \
  --n_ctx "${CTX_SIZE:-8192}" \
  --n_gpu_layers "${N_GPU_LAYERS:-999}" \
  --n_threads "${THREADS:-8}" \
  > logs/llama.log 2>&1 &

# Start Open WebUI on :8998
DATA_DIR=/workspace/open-webui \
WEBUI_AUTH=False \
OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=${OPENAI_API_KEY:-unused} \
DEFAULT_MODELS=${DEFAULT_MODELS:-/workspace/models/${MODEL_FILE}} \
AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST=${AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST:-120} \
nohup .venv/bin/open-webui serve --host 0.0.0.0 --port 8998 > logs/open-webui.log 2>&1 &
```

Check status:

```bash
ss -ltnp | grep -E ':8080|:8998'
curl -sS http://127.0.0.1:8080/v1/models
curl -I http://127.0.0.1:8998
```

Open WebUI:
- `http://<your-server-ip>:8998`

llama.cpp API (direct):
- `http://<your-server-ip>:8080/v1`

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

Then restart:

```bash
docker compose down
docker compose up -d
```

For no-Docker/manual restarts:

```bash
python3 scripts/run_manual.py --skip-install
```
