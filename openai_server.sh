#! /bin/bash
port=8000
max_model_len=2048
max_num_seqs=128
enable_lora=false
prefix_caching=false
model_dir=/home/irteamsu/models
model_name=Meta-Llama-3.1-8B-Instruct
lora_name=Gaudi_LoRA_Llama-3-8B-Instruct
kv_fp8=false
enforce_eager=false
quantization_param_path=/home/jovyan/vol-1/jh/vllm-0.5.4-ma/examples/fp8/ll3-1_8b_inst_fp8_kv/kv_cache_scales.json
num_scheduler_steps=1

error() {
    echo "Error: Invalid option '$1'"
    echo "Usage: $0 [--port <port>] [--max-model-len <length>] [--enable-lora] [--prefix-caching] [--kv-fp8] [--enforce-eager] [--num-scheduler-steps]"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-model-len) max_model_len="$2"; shift ;;
        --max-num-seqs) max_num_seqs="$2"; shift ;;
        --num-scheduler-steps) num_scheduler_steps="$2"; shift ;;
        --enable-lora) enable_lora=true ;;
        --prefix-caching) prefix_caching=true ;;
        --kv-fp8) kv_fp8=true ;;
        --enforce-eager) enforce_eager=true ;;
        *) error "$1" ;;
    esac
    shift
done

# Build LoRA options conditionally
lora_options=""
if [[ "$enable_lora" == "true" ]]; then
    lora_options="--enable-lora --max-loras 2 --max-lora-rank 8 --lora-modules lora-1=$model_dir/$lora_name lora-2=$model_dir/$lora_name"
fi

# Construct the final command
python3 -m vllm.entrypoints.openai.api_server --port $port --max-model-len $max_model_len --model $model_dir/$model_name \
--max-num-seqs $max_num_seqs --disable-log-requests \
--num-scheduler-steps $num_scheduler_steps \
$( [[ "$prefix_caching" == "true" ]] && echo "--enable-prefix-caching" ) \
$( [[ "$kv_fp8" == "true" ]] && echo "--kv-cache-dtype fp8 --quantization-param-path $quantization_param_path" ) \
$( [[ "$enforce_eager" == "true" ]] && echo "--enforce-eager" ) \
$lora_options

