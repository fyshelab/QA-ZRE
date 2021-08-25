#!/bin/bash

python src/question_response_generation/train.py \
    --question_training True --mode squad_test \
    --model_path $MODEL_PATH --learning_rate 0.001 \
    --max_epochs 6 --batch_size 64 \
    --gpu True --gpu_device 0 \
    --checkpoint _0_model
    --prediction_file "./squad_dev.csv"
