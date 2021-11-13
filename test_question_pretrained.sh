#!/bin/bash

source env/bin/activate

printf "Test the pre-trained question generator on the dev set of the RE dataset.\r\n"

main_path=$HOME/t5-small-exps/naacl-2022/posterier_question_model/

for (( i=0; i<=3; i++))
do
	for (( j=1; j<=8; j++ ))
	do
		step=$((j * 100))
		printf "step ${step} on epoch ${i}\r\n"
		python src/re_gold_qa_train.py \
			--mode re_qa_test \
			--model_path ${main_path} \
			--answer_checkpoint _response_pretrained \
			--question_checkpoint _${i}_step_${step}_model \
			--batch_size 64 --gpu True \
			--ignore_unknowns True \
			--train zero-shot-extraction/relation_splits/train.very_small.0 \
			--dev zero-shot-extraction/relation_splits/dev.2 \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file ${main_path}fold_3.question_pretrained_with_best_response_pretrained.dev.predictions.${i}.step.${step}.csv
	done

	printf "full epoch ${i}\r\n"
	python src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path ${main_path} \
		--answer_checkpoint _response_pretrained \
		--question_checkpoint _${i}_model \
		--batch_size 64 --gpu True \
		--ignore_unknowns True \
		--train zero-shot-extraction/relation_splits/train.very_small.0 \
		--dev zero-shot-extraction/relation_splits/dev.2 \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file ${main_path}fold_3.question_pretrained_with_best_response_pretrained.dev.predictions.${i}.step.full.csv

done
