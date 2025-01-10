#!/bin/bash
envs=("lle-lvl6")
configs=("maser-cnn")
n_seeds=20
n_concurrent=8

j=0
for env in ${envs[@]}; do
    for config in ${configs[@]}; do
        for seed in $(seq 0 ${n_seeds}); do
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