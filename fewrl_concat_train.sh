#!/bin/bash

seeds=(12321 943 111 300 1300 545 12 10001 77 1993)
gpu_ids=(0 0 0 1 1 1 2 2 3 3)
for i in ${!seeds[@]};
do
	seed=${seeds[$i]}
	cuda_gpu=${gpu_ids[$i]}
	CUDA_VISIBLE_DEVICES=${cuda_gpu} python3.7 src/re_gold_qa_train.py \
		--mode concat_fewrl_train \
		--model_path ~/sep-1/fewrel/concat_run_${seed}/ \
		--checkpoint _response_pretrained \
		--learning_rate 0.0005 \
		--max_epochs 10 \
		--batch_size 4 \
		--gpu True \
		--train ./fewrl_data/train_data_${seed}.csv \
		--dev ./fewrl_data/val_data_${seed}.csv \
		--test ./fewrl_data/test_data_${seed}.csv \
		--gpu_device 0 \
		--seed ${seed} &
done