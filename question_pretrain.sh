#!/bin/bash

source env/bin/activate

python src/question_response_generation/train.py \
  --mode question_generation_pretrain \
  --model_path $HOME/posterier_question_model/ \
  --learning_rate 0.0005 \
  --max_epochs 4 \
  --batch_size 64 \
  --gpu True \
  --gpu_device 0 \
  --seed 12321
