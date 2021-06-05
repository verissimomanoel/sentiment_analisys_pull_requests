#!/bin/bash

PARAMS=""

if [ "$EARLY_STOPPING" != "" ]
then
    PARAMS="$PARAMS --early-stopping=$EARLY_STOPPING"
fi

if [ "$EPOCHS" != "" ]
then
    PARAMS="$PARAMS --epochs=$EPOCHS"
fi

if [ "$MAX_LEN" != "" ]
then
    PARAMS="$PARAMS --max-len=$MAX_LEN"
fi

if [ "$BATCH_SIZE" != "" ]
then
    PARAMS="$PARAMS --batch_size=$BATCH_SIZE"
fi

if [ "$NUM_WORKERS" != "" ]
then
    PARAMS="$PARAMS --num-workers=$NUM_WORKERS"
fi

if [ "$BASELINE_PATH" != "" ]
then
    PARAMS="$PARAMS --baseline-path=$BASELINE_PATH"
fi

if [ "$LEARNING_RATE" != "" ]
then
    PARAMS="$PARAMS --leaning-rate=$LEARNING_RATE"
fi

python3 main.py --number-of-classes=$NUMBER_OF_CLASSES --train-path-file=$TRAIN_PATH_FILE --val-path-file=$VAL_PATH_FILE --test-path-file=$TEST_PATH_FILE --feature-name=$FEATURE_NAME --target-name=$TARGET_NAME --baseline-path=$BASELINE_PATH --checkpoint-path=$CHECKPOINT_PATH $PARAMS

exec "$@"