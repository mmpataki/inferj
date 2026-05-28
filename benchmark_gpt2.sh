#!/usr/bin/env bash
set -euo pipefail

INFERJ_DIR="${INFERJ_DIR:-$(pwd)}"
LLAMA_DIR="${LLAMA_DIR:-/home/mpataki/projects/llama.cpp}"
JAVA_BIN="${JAVA_BIN:-java}"
LLAMA_CLI="${LLAMA_CLI:-$LLAMA_DIR/build/bin/llama-cli}"
PROMPT="${PROMPT:-once upon a time there}"
GEN_TOKENS="${GEN_TOKENS:-10}"
TEMP="${TEMP:-1.3}"
TOP_K="${TOP_K:-5}"
MODEL_DIR="${MODEL_DIR:-$INFERJ_DIR}"
LLAMA_MODEL="${LLAMA_MODEL:-$LLAMA_DIR/models/GPT-2-f32.gguf}"

require_file() {
  local path="$1"

  if [[ ! -e "$path" ]]; then
    echo "error: missing $path" >&2
    exit 1
  fi
}

extract_last_match() {
  local pattern="$1"
  local input="$2"

  printf '%s\n' "$input" | sed -n "$pattern" | tail -n 1
}

require_file "$INFERJ_DIR/InferJ.class"
require_file "$INFERJ_DIR/model.safetensors"
require_file "$INFERJ_DIR/config.json"
require_file "$INFERJ_DIR/merges.txt"
require_file "$INFERJ_DIR/vocab.json"
require_file "$LLAMA_CLI"
require_file "$LLAMA_MODEL"

inferj_output="$("$JAVA_BIN" InferJ \
  --model-dir "$MODEL_DIR" \
  --prompt "$PROMPT" \
  --max-tokens "$GEN_TOKENS" \
  --temp "$TEMP" \
  --top-k "$TOP_K" 2>&1)"
inferj_toks_per_s="$(extract_last_match 's/.*\[\([0-9.][0-9.]*\) tok\/s\].*/\1/p' "$inferj_output")"

llama_output="$("$LLAMA_CLI" \
  -m "$LLAMA_MODEL" \
  -p "$PROMPT" \
  -n "$GEN_TOKENS" \
  -ngl 0 \
  --simple-io \
  --no-display-prompt \
  --single-turn \
  --temp "$TEMP" \
  --top-k "$TOP_K" 2>&1)"
llama_gen_tps="$(extract_last_match 's/.*Generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$llama_output")"
llama_prompt_tps="$(extract_last_match 's/.*Prompt: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$llama_output")"

cat <<EOF
| Project | Model | Prompt | Gen tokens | Throughput | Notes |
| --- | --- | --- | --- | --- | --- |
| InferJ | model.safetensors GPT-2 FP32 | \`$PROMPT\` | $GEN_TOKENS | \`$inferj_toks_per_s tok/s\` | \`$INFERJ_DIR\` |
| llama.cpp | GPT-2-f32.gguf | \`$PROMPT\` | $GEN_TOKENS | \`$llama_gen_tps tok/s\` | prompt: \`$llama_prompt_tps t/s\`, model: \`$LLAMA_MODEL\` |
EOF
