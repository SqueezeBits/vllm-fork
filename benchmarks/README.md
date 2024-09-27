# SQZB VLLM benchmarking script

We've been conducting benchmarks via `vllm/entrypoints/openai/api_server.py` and `benchmarks/benchmark_sqzb.py`.

## Disclaimer
* Time spent on request queueing is not included in reported TTFT by the benchmarking script.
* We assume prompts in the datasets are already tokenized to exclude input tokenization from the measurement. 

## Benchmarking process

### Benchmarking with dataset
Benchmarking is done in the following 3 steps. Command line examples are for benchmarking a llama3 8B model with a dynamic dataset with 1k input and 1k output.

1. Run end-to-end benchmark

    - Run api server
        ```bash
        python -m vllm.entrypoints.openai.api_server \
            --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
            --block-size 128 \
            --max-model-len 2048 \
            --enforce-eager \
            --disable-log-requests
        ```

    - Send requests
        ```bash
        python benchmarks/benchmark_sqzb.py \
            --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
            --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
            --max-input-len 1024 \
            --max-output-len 1024
        ```

2. Run prefill benchmark

    - Run api server again with the same configuration as step 1
        ```bash
        python -m vllm.entrypoints.openai.api_server \
            --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
            --block-size 128 \
            --max-model-len 2048 \
            --enforce-eager \
            --disable-log-requests
        ```

    - Send requests with max output len set to 1
        ```bash
        python benchmarks/benchmark_sqzb.py \
            --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
            --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
            --max-input-len 1024 \
            --max-output-len 1
        ```

3. Summarize results

Total input tokens, total generated tokens, end-to-end latency, TTFT, TPOT, mean running batch size(average scheduled batch size in decode steps) are obtained via step 1. Prefill latency is obtained in step2. Other metrics are calculated as the following:
* generation latency = end-to-end latency - prefill latency
* prefill throughput = total input tokens / prefill latency
* end-to-end throughput = total generated tokens / end-to-end latency
* generation throughput = total generated tokens / generation latency

### Benchmarking with fixed-length random data

Repeat the steps above from 1 to 3 but with `--dataset` argument omitted in request sending script call. For example:
    ```bash
    python benchmarks/benchmark_sqzb.py \
        --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --max-input-len 1024 \
        --max-output-len 1024
    ```


## Additional Features
Benchmark script now supports additional features including batched multi-LoRA, guided json, automatic prefix caching and FP8 KV cache. Command line examples are as follows:

### A. Batched Multi-LoRA
Currently, Multi-LoRA can be tested under limited configuration(`max_num_seqs` <= 128, `max_num_batched_tokens` == `max_num_seqs` * `max_model_len`) due to [vllm-fork internal bug](https://github.com/HabanaAI/vllm-fork/issues/237).
1. Run api server with LoRA support
    ```bash
    VLLM_PROMPT_BS_BUCKET_MAX=128 \
    python -m vllm.entrypoints.openai.api_server \
        --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --block-size 128 \
        --max-model-len 2048 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 262144 \
        --enable-lora \
        --lora-modules lora-1=/scratch-1/models/Gaudi_LoRA_Llama-3-8B-Instruct lora-2=/scratch-1/models/Gaudi_LoRA_Llama-3-8B-Instruct \
        --max-loras 2 \
        --max-lora-rank 8 \
        --enforce-eager \
        --disable-log-requests
    ```

2. Send requests with LoRA support
    ```bash
    python benchmarks/benchmark_sqzb.py \
        --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
        --max-input-len 1024 \
        --max-output-len 1024 \
        --lora-pattern ,lora-1,lora-2
    ```

### B. Guided JSON
1. Run api server
    ```bash
    python -m vllm.entrypoints.openai.api_server \
        --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --block-size 128 \
        --max-model-len 2048 \
        --enforce-eager \
        --disable-log-requests
    ```

2. Send requests with json template
    ```bash
    python benchmarks/benchmark_sqzb.py \
        --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
        --max-input-len 1024 \
        --max-output-len 1024 \
        --json-template benchmarks/guided_json_template.json
    ```

### C. Automatic Prefix Caching
1. Run api server
    ```bash
    python -m vllm.entrypoints.openai.api_server \
        --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --block-size 128 \
        --max-model-len 2048 \
        --enforce-eager \
        --enable-prefix-caching \
        --disable-log-requests
    ```

2. Send requests
    ```bash
    python benchmarks/benchmark_sqzb.py \
        --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
        --max-input-len 1024 \
        --max-output-len 1024
    ```

### D. FP8 KV Cache
0. Install INC from modified source.
    ```bash
    git submodule update --init --recursive
    pushd neural-compressor
    pip install -e .
    python setup.py develop pt
    popd
    ```

1. Perform INC measurement mode with benchmark_throughput script.
    ```bash
    QUANT_CONFIG=configs/measure.json QUANT_VERBOSE=1 VLLM_SKIP_WARMUP=True python benchmarks/benchmark_throughput.py \
        --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --input-len 1024 \
        --output-len 1024 \
        --num-prompts 128 \
        --quantization inc
    ```
    Check whether the measurement results are properly generated under artifacts/ directory.

2. Perform benchmark with quantization
    1. Run api server
        ```bash
        QUANT_CONFIG=configs/quantize.json QUANT_VERBOSE=1 python -m vllm.entrypoints.openai.api_server \
            --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
            --block-size 128 \
            --max-model-len 2048 \
            --enforce-eager \
            --quantization inc \
            --kv-cache-dtype fp8_inc \
            --disable-log-requests
        ```

    2. Send requests
        ```bash
        python benchmarks/benchmark_sqzb.py \
            --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
            --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
            --max-input-len 1024 \
            --max-output-len 1024
        ```

### E. Benchmarking with all features in interest enabled
0. Install INC from modified source.
    ```bash
    git submodule update --init --recursive
    pushd neural-compressor
    pip install -e .
    python setup.py develop pt
    popd
    ```

1. Perform INC measurement mode with benchmark_throughput script.
    ```bash
    QUANT_CONFIG=configs/measure.json QUANT_VERBOSE=1 VLLM_SKIP_WARMUP=True python benchmarks/benchmark_throughput.py \
        --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --input-len 1024 \
        --output-len 1024 \
        --num-prompts 128 \
        --quantization inc
    ```
    Check whether the measurement results are properly generated under artifacts/ directory.

2. Run api server
    ```bash
    QUANT_CONFIG=configs/quantize.json QUANT_VERBOSE=1 \
    VLLM_PROMPT_BS_BUCKET_MAX=128 \
    python -m vllm.entrypoints.openai.api_server \
        --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --block-size 128 \
        --max-model-len 2048 \
        --enforce-eager \
        --max-num-seqs 128 \
        --max-num-batched-tokens 262144 \
        --enable-lora \
        --lora-modules lora-1=/scratch-1/models/Gaudi_LoRA_Llama-3-8B-Instruct lora-2=/scratch-1/models/Gaudi_LoRA_Llama-3-8B-Instruct \
        --max-loras 2 \
        --max-lora-rank 8 \
        --enable-prefix-caching \
        --quantization inc \
        --kv-cache-dtype fp8_inc \
        --disable-log-requests
    ```

3. Send requests
    ```bash
    python benchmarks/benchmark_sqzb.py \
        --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
        --dataset /scratch-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_256_max_1024_1024_sampled.parquet \
        --max-input-len 1024 \
        --max-output-len 1024 \
        --lora-pattern ,lora-1,lora-2 \
        --json-template benchmarks/guided_json_template.json
    ```
