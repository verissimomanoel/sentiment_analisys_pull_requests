#!/bin/bash

python3 main.py --baseline-path="./baseline" --checkpoint-path="./model" --train-path-file="./data/train.csv" --val-path-file="./data/val.csv" --test-path-file="./data/test.csv" --number-of-classes=3 --feature-name="text" --target-name="airline_sentiment" --early-stopping=3

exec "$@"