#!/bin/bash
envs=("lle")
configs=("haven-cnn-vdn-no-ir")
n_seeds=8
n_concurrent=16

j=0
for env in ${envs[@]}; do
    for config in ${configs[@]}; do
        for seed in $(seq 0 $(($n_seeds-1))); do
            echo "Starting job number ${j}: ${config} ${env} seed ${seed}"
            python src/main.py --config=${config} --env-config=${env} with env_args.seed=${seed} &
            if [ $(jobs -p | wc -l) -ge ${n_concurrent} ]; then
                wait -n
            fi
            j=$((j+1))
        done
    done
done


echo "Finished launching jobs"