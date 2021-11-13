#!/bin/bash

source env/bin/activate

python src/question_response_generation/train.py \
    --mode all_train \
    --model_path $HOME/response_pretrained_model/ \
    --learning_rate 0.0005 
    --max_epochs 4 \
    --batch_size 64  \
    --gpu True \
    --gpu_device 0 \
    --seed 12321