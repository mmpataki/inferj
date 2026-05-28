#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-gpt2}"
REVISION="${REVISION:-main}"
DEST_DIR="${1:-.}"
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/${REVISION}"

files=(
  "config.json"
  "merges.txt"
  "vocab.json"
  "model.safetensors"
)

mkdir -p "$DEST_DIR"

download() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl --location --fail --retry 3 --continue-at - --output "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --tries=3 --output-document="$out" "$url"
  else
    echo "error: install curl or wget first" >&2
    return 1
  fi
}

for file in "${files[@]}"; do
  target="${DEST_DIR%/}/$file"
  tmp="${target}.part"

  if [[ -s "$target" ]]; then
    echo "ok: $target already exists"
    continue
  fi

  echo "downloading: $file"
  download "${BASE_URL}/${file}" "$tmp"

  if [[ ! -s "$tmp" ]]; then
    echo "error: downloaded file is empty: $file" >&2
    exit 1
  fi

  mv "$tmp" "$target"
done

echo "done: GPT-2 safetensors files are in $DEST_DIR"
