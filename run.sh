#!/bin/bash

for i in {0..15}; do
    python src/main.py --env-config=lle --config=haven-cnn with env_args.seed=${i} &
done

