#!/bin/bash
set -e

# Load environment from .env
set -a
source .env
set +a

echo "=== Checking vLLM at ${VLLM_BASE_URL} ==="
until curl -sf "${VLLM_BASE_URL}/models" > /dev/null 2>&1; do
    echo "vLLM not ready, waiting 15s..."
    sleep 15
done
echo "vLLM is ready!"
curl -s "${VLLM_BASE_URL}/models" | python3 -m json.tool

echo ""
echo "=== Running AIR-BENCH-2024 ==="
python3 run_benchmark.py \
    --vllm-base-url "${VLLM_BASE_URL}" \
    --vllm-model "${VLLM_MODEL}" \
    --anthropic-api-key "${ANTHROPIC_API_KEY}" \
    --sample-num 5 \
    --output-dir results
