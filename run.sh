#!/bin/bash
envs=("lle-lvl6")
configs=("vdn-cnn" "qmix-cnn" "haven-cnn-vdn" "haven-cnn-qmix")
n_seeds=20
n_concurrent=16

j=0
for env in ${envs[@]}; do
    for config in ${configs[@]}; do
        for i in $(seq 20 40); do
            echo "Starting job number ${j}: ${config} ${env} seed ${i}"
            python src/main.py --config=${config} --env-config=${env} with env_args.seed=${i} &
            if [ $(jobs -p | wc -l) -ge ${n_concurrent} ]; then
                wait -n
            fi
            j=$((j+1))
        done
    done
done


echo "Finished launching jobs"