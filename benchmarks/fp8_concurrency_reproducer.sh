# 1HPU
bash run_fp8.sh --port 8080 --postfix 11
wait
sleep 30

# 2HPUs
bash run_fp8.sh --port 8080 --postfix 21 &
bash run_fp8.sh --port 8181 --postfix 22 &
wait
sleep 30

# 4HPUs
bash run_fp8.sh --port 8080 --postfix 41 &
bash run_fp8.sh --port 8181 --postfix 42 &
bash run_fp8.sh --port 8282 --postfix 43 &
bash run_fp8.sh --port 8383 --postfix 44 &
wait
sleep 30

# 8HPUs
bash run_fp8.sh --port 8080 --postfix 81 &
bash run_fp8.sh --port 8181 --postfix 82 &
bash run_fp8.sh --port 8282 --postfix 83 &
bash run_fp8.sh --port 8383 --postfix 84 &
bash run_fp8.sh --port 8484 --postfix 85 &
bash run_fp8.sh --port 8585 --postfix 86 &
bash run_fp8.sh --port 8686 --postfix 87 &
bash run_fp8.sh --port 8787 --postfix 88 &
wait
