#!/bin/bash
#configs[0]="--config=maser-cnn --env-config=lle-lvl6"
#configs[0]="--config=qmix-cnn --env-config=shaped-lle"
configs[0]="--config=qplex --env-config=lle-lvl6"
configs[1]="--config=qplex --env-config=shaped-lle"
n_seeds=8
n_concurrent=16

j=0
for config in "${configs[@]}"; do
    for seed in $(seq 0 $(($n_seeds-1))); do
        echo "Starting job number ${j}: ${config} with seed=${seed}"
        python src/main.py ${config} with env_args.seed=${seed} &
        if [ $(jobs -p | wc -l) -ge ${n_concurrent} ]; then
            wait -n
        fi
        j=$((j+1))
    done
done


echo "Finished launching jobs"