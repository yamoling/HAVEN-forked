#!/bin/bash
envs=("shaped-lle-lvl6")
configs=("vdn-cnn" "vdn-cnn-no-ir" "haven-cnn-vdn" "haven-cnn-vdn-no-ir" "haven-cnn-qmix" "haven-cnn-qmix-no-ir" "qmix-cnn" "qmix-cnn-no-ir")
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