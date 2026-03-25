#!/usr/bin/env bash

MODEL_NAME="${1:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${2:-8000}"

vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto \
  --api-key demo-key