# InferJ

InferJ is a small Java inference engine for GPT-2. It loads GPT-2 weights
from `model.safetensors`, reads the Hugging Face tokenizer files, and runs CPU
generation without external Java dependencies.

The project is intentionally compact: the tensor operations, JSON parsing,
safetensors loading, tokenizer, and GPT-2 forward pass all live in
`InferJ.java`.

## What It Supports

- GPT-2 style transformer inference
- Hugging Face `model.safetensors` weights
- `config.json`, `merges.txt`, and `vocab.json`
- Byte-pair encoding tokenization
- Temperature-scaled top-k multinomial sampling
- A lightweight tensor implementation for matmul, softmax, layer norm, GELU,
  slicing, transpose, view, split, and broadcasting

## Current Limits

- CPU only
- Batch size 1
- No KV cache
- The safetensors loader assumes FP32 tensors
- `--top-p` is not supported yet

## Download GPT-2 Files

Use the included helper script to download the required GPT-2 files:

```bash
./pull_gpt2_safetensors.sh
```

That downloads:

```text
config.json
merges.txt
vocab.json
model.safetensors
```

You can also choose another destination:

```bash
./pull_gpt2_safetensors.sh /path/to/gpt-2
```

## Build

Compile with a recent JDK:

```bash
javac InferJ.java TensorTests.java
```

## Run

`InferJ.main` supports these flags:

- `--model-dir` default: `.`
- `--prompt` default: `once upon a time there`
- `--max-tokens` default: `100`
- `--temp` default: `1.3`
- `--top-k` default: `5`

Example:

```bash
java InferJ \
  --model-dir /path/to/gpt-2 \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temp 1.3 \
  --top-k 5
```

The program prints the generated text as it samples tokens and reports the
running token rate.

## Benchmarks

Baseline comparison on the local machine using the same prompt and a 10-token
generation window:

```bash
./benchmark_gpt2.sh
```

You can override the defaults with environment variables such as `PROMPT`,
`GEN_TOKENS`, `TEMP`, `TOP_K`, `INFERJ_DIR`, `LLAMA_DIR`, and `LLAMA_MODEL`.

| Project | Prompt | Gen tokens | Throughput | Notes |
| --- | --- | --- | --- | --- |
| InferJ | `once upon a time there` | 10 | `0.254130 tok/s` | Current Java implementation in this repo |
| llama.cpp | `once upon a time there` | 10 | `31.7 tok/s` | `llama-cli` on CPU reported `Prompt: 646.7 t/s` and `Generation: 31.7 t/s` |

Update this table as the implementation improves.

## Tests

The tensor layer has fuzz-style tests in `TensorTests.java`:

```bash
java TensorTests
```

These tests exercise tensor layout behavior and core ops used by the GPT-2
forward pass.

## Project Layout

```text
InferJ.java                  Inference engine, tensor ops, tokenizer, loaders
TensorTests.java             Tensor operation tests
benchmark_gpt2.sh            GPT-2 throughput comparison helper
pull_gpt2_safetensors.sh     GPT-2 safetensors download helper
```
