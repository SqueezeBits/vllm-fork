# SQZB VLLM benchmarking script

We've been conducting benchmarks via `vllm/entrypoints/openai/api_server.py` and `benchmarks/benchmark_sqzb.py`.

## Disclaimer
* Time spent on request queueing is not measured.
* We assume prompts in the datasets are already tokenized to exclude input tokenization from the measurement. 

## Benchmarking process
Benchmarking is done in following 5 steps. Command line examples are for benchmarking a llama3 8B model with a dynamic dataset with 1k input and 1k output with prefix caching enabled.

1. Run openai server
```
python -m vllm.entrypoints.openai.api_server \
    --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --block-size 128 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --enforce-eager
```

2. Send requests
```
python benchmarks/benchmark_sqzb.py \
    --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --dataset /scratch-1/datasets/llama3-1k.pkl \
    --max-input-len 1024 \
    --max-output-len 1024
```

3. Run openai server again with the same configuration as step 1
```
python -m vllm.entrypoints.openai.api_server \
    --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --block-size 128 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --enforce-eager
```

4. Send requests with the same configuration as step 2 but max output len set to 1
```
python benchmarks/benchmark_sqzb.py \
    --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --dataset /scratch-1/datasets/llama3-1k.pkl \
    --max-input-len 1024 \
    --max-output-len 1
```

5. Summarize results

Total input tokens, total generated tokens, end-to-end latency, TTFT, TPOT are obtained via step 2. Summarization latency is obtained in step4. Other metrics are calculated as the following:
* generation latency = end-to-end latency - summarization latency
* summarization throughput = total input tokens / summarization latency
* end-to-end throughput = total generated tokens / end-to-end latency
* generation throughput = total generated tokens / generation latency


## Additional Features
Benchmark script now supports additional features including batched multi-LoRA and guided json. Command line examples are as follows:

### Batched Multi-LoRA
1. Run openai server with LoRA support
```
python -m vllm.entrypoints.openai.api_server \
    --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --block-size 128 \
    --max-model-len 2048 \
    --max-loras 2 \
    --max-lora-rank 8 \
    --lora-modules lora-1=/scratch-1/models/Gaudi_LoRA_Llama-3-8B-Instruct lora-2=/scratch-1/models/Gaudi_LoRA_Llama-3-8B-Instruct \
    --enforce-eager
```

2. Send requests with LoRA support
```
python benchmarks/benchmark_sqzb.py \
    --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --dataset /scratch-1/datasets/llama3-1k.pkl \
    --max-input-len 1024 \
    --max-output-len 1024 \
    --enable-lora \
    --lora-pattern ,lora-1,lora-2 \
    --random-lora
```

### Guided JSON
1. Run openai server
```
python -m vllm.entrypoints.openai.api_server \
    --model /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --block-size 128 \
    --max-model-len 2048 \
    --enforce-eager
```

2. Send requests with json template
```
python benchmarks/benchmark_sqzb.py \
    --tokenizer /scratch-1/models/Meta-Llama-3-8B-Instruct \
    --dataset /scratch-1/datasets/llama3-1k.pkl \
    --max-input-len 1024 \
    --max-output-len 1024 \
    --guided-json \
    --json-template benchmarks/guided_json_template.json
```
