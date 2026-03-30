#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required."
  exit 1
fi

if [ ! -d "${ROOT_DIR}/.git" ]; then
  echo "No .git directory found at ${ROOT_DIR}."
  exit 1
fi

cd "${ROOT_DIR}"

git add .gitignore README.md .env.example docker-compose.yml requirements.runpod.txt scripts/

echo "\nStaged minimal reproducible files:"
git diff --cached --name-only

echo "\nNext steps:"
echo "  1) git commit -m 'Minimal reproducible RunPod setup'"
echo "  2) git push origin <branch>"
