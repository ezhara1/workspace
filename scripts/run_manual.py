#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
VENV = ROOT / ".venv"
VIBEVOICE_VENV = ROOT / ".venv-vibevoice"
REQUIREMENTS_FILE = ROOT / "requirements.runpod.txt"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"
WEBUI_DATA_DIR = ROOT / "open-webui"
VIBEVOICE_API_SCRIPT = ROOT / "scripts" / "vibevoice_openai_tts_api.py"


def run(cmd: list[str], check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), check=check, text=True, capture_output=False, env=env)


def run_capture(cmd: list[str], env: dict[str, str] | None = None) -> str:
    print("+", " ".join(cmd))
    out = subprocess.check_output(cmd, cwd=str(ROOT), text=True, env=env)
    return out


def parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v
    return env


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(updates)
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key, _ = line.split("=", 1)
        if key in remaining:
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)

    for key, value in remaining.items():
        output.append(f"{key}={value}")

    path.write_text("\n".join(output).rstrip() + "\n")


def ensure_model(env: dict[str, str]) -> str:
    model_repo = env.get("MODEL_REPO", "").strip()
    model_file = env.get("MODEL_FILE", "").strip()
    hf_token = os.environ.get("HF_TOKEN", env.get("HF_TOKEN", "")).strip()

    if not model_repo:
        raise RuntimeError("MODEL_REPO is empty in .env")

    if model_file and (MODELS_DIR / model_file).exists():
        return model_file

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_gguf.py"),
        "--repo",
        model_repo,
        "--out-dir",
        str(MODELS_DIR),
    ]
    if model_file:
        cmd += ["--file", model_file]
    if hf_token:
        cmd += ["--token", hf_token]

    output = run_capture(cmd)
    discovered = output.strip().splitlines()[-1].strip() if output.strip() else ""
    if not discovered:
        raise RuntimeError("Failed to discover/download GGUF model")
    return discovered


def download_model_from_url(model_url: str, file_name: str | None = None) -> str:
    parsed = urllib.parse.urlparse(model_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("--model-url must be an http(s) URL")

    candidate_name = file_name or os.path.basename(urllib.parse.unquote(parsed.path))
    if not candidate_name:
        raise RuntimeError("Could not infer model filename from URL path. Use --model-file-name.")
    if not candidate_name.lower().endswith(".gguf"):
        raise RuntimeError("Model URL filename must end with .gguf (or pass --model-file-name <name>.gguf)")

    dest = MODELS_DIR / candidate_name
    if dest.exists():
        print(f"Model already exists: {dest}")
        return candidate_name

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading model from URL: {model_url}")
    print(f"Saving to: {dest}")
    req = urllib.request.Request(model_url)
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100
                print(f"\rDownloading: {pct:5.1f}%", end="", flush=True)
    if total:
        print()

    tmp.replace(dest)
    print(f"Saved model: {dest}")
    return candidate_name


def ensure_python_runtime(skip_install: bool) -> None:
    if not VENV.exists():
        run([sys.executable, "-m", "venv", str(VENV)])

    pip = str(VENV / "bin" / "pip")
    if skip_install:
        return

    run([pip, "install", "--upgrade", "pip", "setuptools", "wheel"])
    if REQUIREMENTS_FILE.exists():
        run([pip, "install", "-r", str(REQUIREMENTS_FILE)])
    else:
        run([
            pip,
            "install",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cu124",
            "llama-cpp-python[server]",
        ])
        run([pip, "install", "open-webui"])


def ensure_vibevoice_runtime(skip_install: bool) -> None:
    created = False
    if not VIBEVOICE_VENV.exists():
        run([sys.executable, "-m", "venv", str(VIBEVOICE_VENV)])
        created = True

    if skip_install and not created:
        return

    pip = str(VIBEVOICE_VENV / "bin" / "pip")
    run([pip, "install", "-U", "pip"])
    run(
        [
            pip,
            "install",
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    )
    run(
        [
            pip,
            "install",
            "vibevoice[streamingtts] @ git+https://github.com/microsoft/VibeVoice.git",
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "requests",
            "numpy",
        ]
    )


def stop_existing() -> None:
    run(["pkill", "-f", "llama_cpp.server"], check=False)
    run(["pkill", "-f", "open-webui serve"], check=False)
    run(["pkill", "-f", "vibevoice_openai_tts_api.py"], check=False)


def start_services(env: dict[str, str], webui_auth: bool, tts_port: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    python_bin = str(VENV / "bin" / "python")
    open_webui_bin = str(VENV / "bin" / "open-webui")

    model_file = env["MODEL_FILE"]
    ctx_size = env.get("CTX_SIZE", "8192")
    n_gpu_layers = env.get("N_GPU_LAYERS", "999")
    threads = env.get("THREADS", "8")
    enable_thinking = env.get("ENABLE_THINKING", "true").strip().lower() in {"1", "true", "yes", "on"}
    llama_extra_args = env.get("LLAMA_SERVER_EXTRA_ARGS", "").strip()
    openai_api_key = env.get("OPENAI_API_KEY", "unused")
    default_models = env.get("DEFAULT_MODELS", f"/workspace/models/{model_file}")
    model_timeout = env.get("AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST", "120")
    vibevoice_tts_model = env.get("VIBEVOICE_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
    vibevoice_strip_think = env.get("VIBEVOICE_STRIP_THINK_FOR_TTS", "true")

    llama_cmd = [
        python_bin,
        "-m",
        "llama_cpp.server",
        "--model",
        str(MODELS_DIR / model_file),
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--n_ctx",
        ctx_size,
        "--n_gpu_layers",
        n_gpu_layers,
        "--n_threads",
        threads,
    ]
    if not enable_thinking:
        llama_cmd += [
            "--chat_template_kwargs",
            json.dumps({"enable_thinking": False}, separators=(",", ":")),
        ]
    if llama_extra_args:
        llama_cmd += shlex.split(llama_extra_args)

    webui_env = os.environ.copy()
    webui_env.update(
        {
            "DATA_DIR": str(WEBUI_DATA_DIR),
            "WEBUI_AUTH": "True" if webui_auth else "False",
            "OPENAI_API_BASE_URL": "http://127.0.0.1:8080/v1",
            "OPENAI_API_KEY": openai_api_key,
            "DEFAULT_MODELS": default_models,
            "AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST": model_timeout,
        }
    )

    with (LOGS_DIR / "llama.log").open("ab") as llama_log:
        subprocess.Popen(llama_cmd, cwd=str(ROOT), stdout=llama_log, stderr=subprocess.STDOUT, start_new_session=True)

    with (LOGS_DIR / "open-webui.log").open("ab") as webui_log:
        subprocess.Popen(
            [open_webui_bin, "serve", "--host", "0.0.0.0", "--port", "8998"],
            cwd=str(ROOT),
            stdout=webui_log,
            stderr=subprocess.STDOUT,
            env=webui_env,
            start_new_session=True,
        )

    tts_env = os.environ.copy()
    tts_env.update(
        {
            "VIBEVOICE_API_HOST": "0.0.0.0",
            "VIBEVOICE_API_PORT": tts_port,
            "VIBEVOICE_REQUIRE_CUDA": "true",
            "VIBEVOICE_TTS_MODEL": vibevoice_tts_model,
            "VIBEVOICE_STRIP_THINK_FOR_TTS": vibevoice_strip_think,
        }
    )
    with (LOGS_DIR / "vibevoice-api.log").open("ab") as tts_log:
        subprocess.Popen(
            [str(VIBEVOICE_VENV / "bin" / "python"), str(VIBEVOICE_API_SCRIPT)],
            cwd=str(ROOT),
            stdout=tts_log,
            stderr=subprocess.STDOUT,
            env=tts_env,
            start_new_session=True,
        )


def wait_http(url: str, timeout_sec: int = 60) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full manual (no-Docker) setup and start llama.cpp + Open WebUI + VibeVoice TTS"
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip pip installs in .venv")
    parser.add_argument("--webui-auth", action="store_true", help="Enable Open WebUI auth/login")
    parser.add_argument("--model-url", default="", help="Direct URL to a .gguf model file to download/use")
    parser.add_argument("--model-file-name", default="", help="Optional output filename for --model-url (must end in .gguf)")
    parser.add_argument("--tts-port", default="9001", help="Port for VibeVoice OpenAI-compatible TTS API (default: 9001)")
    args = parser.parse_args()

    if not ENV_FILE.exists():
        if not ENV_EXAMPLE.exists():
            raise RuntimeError("Missing both .env and .env.example")
        shutil.copy2(ENV_EXAMPLE, ENV_FILE)
        print(f"Created {ENV_FILE} from {ENV_EXAMPLE}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WEBUI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = parse_env(ENV_FILE)
    if args.model_url:
        model_file = download_model_from_url(args.model_url, args.model_file_name or None)
    else:
        model_file = ensure_model(env)

    default_model_path = f"/workspace/models/{model_file}"
    updates = {
        "MODEL_FILE": model_file,
        "N_GPU_LAYERS": env.get("N_GPU_LAYERS", "999") or "999",
        "OPENAI_API_BASE_URL": "http://127.0.0.1:8080/v1",
        "ENABLE_THINKING": env.get("ENABLE_THINKING", "true") or "true",
        "AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST": env.get("AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST", "120") or "120",
        "DEFAULT_MODELS": default_model_path if args.model_url else (env.get("DEFAULT_MODELS", default_model_path) or default_model_path),
        "VIBEVOICE_API_PORT": env.get("VIBEVOICE_API_PORT", args.tts_port) or args.tts_port,
        "VIBEVOICE_REQUIRE_CUDA": "true",
        "VIBEVOICE_TTS_MODEL": env.get("VIBEVOICE_TTS_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
        or "microsoft/VibeVoice-Realtime-0.5B",
        "VIBEVOICE_STRIP_THINK_FOR_TTS": env.get("VIBEVOICE_STRIP_THINK_FOR_TTS", "true") or "true",
    }
    if updates["N_GPU_LAYERS"] == "0":
        updates["N_GPU_LAYERS"] = "999"

    upsert_env(ENV_FILE, updates)
    env = parse_env(ENV_FILE)

    ensure_python_runtime(skip_install=args.skip_install)
    ensure_vibevoice_runtime(skip_install=args.skip_install)
    stop_existing()
    start_services(
        env,
        webui_auth=args.webui_auth,
        tts_port=args.tts_port,
    )

    llama_ok = wait_http("http://127.0.0.1:8080/v1/models", timeout_sec=90)
    webui_ok = wait_http("http://127.0.0.1:8998", timeout_sec=90)
    tts_ok = wait_http(f"http://127.0.0.1:{args.tts_port}/health", timeout_sec=180)

    print("\n=== Status ===")
    print("llama.cpp API:", "OK" if llama_ok else "NOT READY")
    print("Open WebUI:", "OK" if webui_ok else "NOT READY")
    print("VibeVoice TTS API:", "OK" if tts_ok else "NOT READY")
    print("Logs:")
    print("  -", LOGS_DIR / "llama.log")
    print("  -", LOGS_DIR / "open-webui.log")
    print("  -", LOGS_DIR / "vibevoice-api.log")
    print("\nEndpoints:")
    print("  - Open WebUI: http://<your-server-ip>:8998")
    print("  - llama.cpp:  http://<your-server-ip>:8080/v1")
    print(f"  - VibeVoice TTS: http://<your-server-ip>:{args.tts_port}/v1")

    status_ok = llama_ok and webui_ok and tts_ok
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
