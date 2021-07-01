#! /bin/bash
n_gpus=$( nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
let max_gpu_ind=$n_gpus-1
file_base=$(mktemp)
start_time=$(date '+%Y%m%d_%H_%M_%S')
pids=""
for i in `seq 0 $max_gpu_ind`; do    
    CUDA_VISIBLE_DEVICES=$i python -m test test_parallel $start_time $file_base $n_gpus ${i} "$@" &
done

wait < <(jobs -p)

python -m test test_parallel $start_time $file_base $n_gpus analysis "$@"
