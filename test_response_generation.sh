#!/bin/bash

source env/bin/activate

python src/response_generation/train.py \
	--mode narrativeqa_test \
	--model_path $HOME/re_response_generation_model/ \
	--learning_rate 0.001 --max_epochs 6 \
	--batch_size 8  --gpu True \
	--gpu_device 0 --prediction_file "narrativeqa_dev.epoch3.csv" \
	--checkpoint _3_model
