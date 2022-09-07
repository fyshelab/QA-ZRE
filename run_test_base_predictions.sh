#!/bin/bash

for (( i=0; i<=9; i++ ))
do
	fold_i=$((i+1))
	CUDA_VISIBLE_DEVICES=1 python3.7 src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path ~/reqa-predictions/ \
		--answer_checkpoint _response_pretrained \
		--question_checkpoint _fold_1_question_pretrained \
		--training_steps 10000 \
		--learning_rate 0.0005 \
		--max_epochs 1 \
		--num_search_samples 8 \
		--batch_size 128 \
		--gpu True \
		--dev ./zero-shot-extraction/relation_splits/test.${i} \
		--train ./zero-shot-extraction/relation_splits/train.${i} \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file ~/reqa-predictions/fold_${fold_i}/base-base.test.predictions.fold.${fold_i}.csv
done