#!/bin/bash
#configs[0]="--config=maser-cnn --env-config=lle-lvl6"
configs[0]="--config=vdn-cnn --env-config=shaped-lle"
n_seeds=8
n_concurrent=8

j=0
for config in "${configs[@]}"; do
    for seed in $(seq 8 15); do
        echo "Starting job number ${j}: ${config} with seed=${seed}"
        python src/main.py ${config} with env_args.seed=${seed} &
        if [ $(jobs -p | wc -l) -ge ${n_concurrent} ]; then
            wait -n
        fi
        j=$((j+1))
    done
done


echo "Finished launching jobs"