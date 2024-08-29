#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

export VLLM_HOST_IP=100.81.254.210
export VLLM_PORT=12345

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# prefilling instance
VLLM_LOGGING_LEVEL=DEBUG VLLM_HOST_IP=100.81.254.210 VLLM_PORT=12345 VLLM_SKIP_WARMUP=true VLLM_RPC_PORT=5570 VLLM_DISAGG_PREFILL_ROLE=prefill python \
    -m vllm.entrypoints.openai.api_server \
    --model /scratch-1/models/Meta-Llama-3-8B-Instruct/ \
    --port 8100 \
    --block-size 128 \
    --max-model-len 2048 \
    --enforce-eager &

# decoding instance
VLLM_LOGGING_LEVEL=DEBUG VLLM_HOST_IP=100.81.254.210 VLLM_PORT=12345 VLLM_SKIP_WARMUP=true VLLM_RPC_PORT=5580 VLLM_DISAGG_PREFILL_ROLE=decode python \
    -m vllm.entrypoints.openai.api_server \
    --model /scratch-1/models/Meta-Llama-3-8B-Instruct/ \
    --port 8200 \
    --block-size 128 \
    --max-model-len 2048 \
    --enforce-eager &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
flask --app disagg_prefill_proxy run -p 12346 &
sleep 3

# serve an example request
curl http://localhost:12346/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/scratch-1/models/Meta-Llama-3-8B-Instruct/",
"prompt": "San Francisco is a",
"max_tokens": 20,
"temperature": 0
}'

# clean up
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9
ps -e | grep flask | awk '{print $1}' | xargs kill -9