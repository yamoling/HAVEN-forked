#!/bin/bash

for i in {0..7}; do
    # python src/main.py --env-config=lle-2lasers --config=haven-cnn with env_args.seed=${i} &
    python src/main.py --env-config=sc2 --config=haven with env_args.seed=${i} &
done

