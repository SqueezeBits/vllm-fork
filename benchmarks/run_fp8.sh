PORT=8080
POSTFIX=""
TIMEOUT=72000
LOG_DIR=logs
MODEL=/home/irteamsu/models/Meta-Llama-3.1-8B-Instruct

mkdir -p ${LOG_DIR}

while [[ "$1" != "" ]]; do
    case "$1" in
        --port)
                shift
                PORT="$1"
                ;;
        --postfix)
                shift
                POSTFIX="$1"
                ;;
    esac
    shift
done


VLLM_SKIP_WARMUP=True QUANT_CONFIG=configs/quantize_all.json QUANT_VERBOSE=1 python -m vllm.entrypoints.openai.api_server \
    --port ${PORT} \
    --model ${MODEL} \
    --block-size 128 \
    --max-model-len 2048 \
    --disable-log-requests \
    --quantization inc \
    --kv-cache-dtype fp8_inc \
    --disable-log-requests \
    &> ${LOG_DIR}/server_${POSTFIX}.log &

pid=$!
start_time=$(date +%s)
port_open=False
while true; do
    nmap_output=$(nmap -p $PORT localhost | grep "$PORT/tcp open")
    if [[ -n "$nmap_output" ]]; then
        port_open=True
        break
    fi
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [[ $elapsed_time -ge $TIMEOUT ]]; then
        echo "Timeout reached. Port $PORT did not open within $TIMEOUT seconds."
        break
    fi
    sleep 5
done

python benchmark_sqzb.py \
    --port ${PORT} \
    --tokenizer ${MODEL} \
    --num-requests 256 \
    --max-input-len 1024 \
    --max-output-len 1024 \
    &> ${LOG_DIR}/client_${POSTFIX}.log

kill -9 $(ps -o pid= --ppid $pid)
kill -9 $pid

