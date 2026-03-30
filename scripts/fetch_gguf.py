#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PREFERRED_PATTERNS = [
    "Q5_K_M",
    "Q4_K_M",
    "Q6_K",
    "Q8_0",
    "Q4_0",
]


def http_get_json(url: str, token: str | None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_gguf_filename(files: list[str]) -> str:
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        raise RuntimeError("No GGUF file found in the repository.")

    for pattern in PREFERRED_PATTERNS:
        for name in ggufs:
            if pattern.lower() in name.lower():
                return name
    return ggufs[0]


def download_file(url: str, output_path: Path, token: str | None) -> None:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req) as resp, output_path.open("wb") as out:
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
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GGUF model file from Hugging Face repo")
    parser.add_argument("--repo", required=True, help="Hugging Face repo, e.g. org/model")
    parser.add_argument("--out-dir", default="models", help="Output directory")
    parser.add_argument("--file", default="", help="Optional GGUF file name to force")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Optional Hugging Face token")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://huggingface.co/api/models/{urllib.parse.quote(args.repo, safe='/')}"
    data = http_get_json(api_url, args.token or None)
    files = [s["rfilename"] for s in data.get("siblings", []) if "rfilename" in s]

    if args.file:
        model_file = args.file
        if model_file not in files:
            raise RuntimeError(f"Requested file '{model_file}' not found in repository.")
    else:
        model_file = pick_gguf_filename(files)

    output_path = out_dir / model_file
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        print(model_file)
        return 0

    encoded = urllib.parse.quote(model_file)
    download_url = f"https://huggingface.co/{args.repo}/resolve/main/{encoded}?download=true"

    print(f"Downloading {model_file} from {args.repo} ...")
    download_file(download_url, output_path, args.token or None)
    print(f"Saved to: {output_path}")

    # stdout is consumed by setup.sh to populate MODEL_FILE
    print(model_file)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
