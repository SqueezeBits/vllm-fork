#!/bin/bash
port=8000
model_name=Meta-Llama-3.1-8B-Instruct
csv_path=/home/irteamsu/works/jongho/benchmark_vllm_v063_1108
max_input_len=1024
steps=1
dynamic_dataset=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        --steps) steps="$2"; shift ;;
        --dynamic-dataset) dynamic_dataset=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

prefix_len=$((max_input_len / 4))
dataset_path=/home/irteamsu/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_${prefix_len}_max_${max_input_len}_1024_sampled.parquet

concurrency_values=(128 128 112 96 80 64 48 32 16)

for concurrency in "${concurrency_values[@]}"; do
    python benchmarks/benchmark_sqzb.py \
    --tokenizer /home/irteamsu/models/Meta-Llama-3.1-8B-Instruct \
    --num-requests 512 \
    --max-input-len $max_input_len \
    --max-output-len 1 \
    --port $port \
    --concurrency $concurrency \
    --csv-path ${csv_path}_${steps}/ \
    $( [[ "$dataset_path" == "true" ]] && echo "--dataset $dataset_path " ) 
done
