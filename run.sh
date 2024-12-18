#!/bin/bash

for i in {0..15}; do
    # python src/main.py --env-config=lle-2lasers --config=haven-cnn with env_args.seed=${i} &
    python src/main.py --config=haven-cnn-vdn --env-config=lle-lvl6 with env_args.seed=${i} &
done

sleep 10h

for i in {0..15}; do
    # python src/main.py --env-config=lle-2lasers --config=haven-cnn with env_args.seed=${i} &
    python src/main.py --config=haven-cnn-qmix --env-config=lle-lvl6 with env_args.seed=${i} &
done

