#!/bin/bash

source env/bin/activate

for (( j=0; j<=109; j++))
do
	k=$((j * 4))
	end_k=$((k+3))
	for (( i=${k}; i<=${end_k}; i++ ))
	do
		step=$(((i+1) * 100))
		printf "step ${step}\r\n"
		python src/re_gold_qa_train.py \
			--mode re_qa_test \
			--model_path $HOME/fold_1/mml-pgg-off-sim/ \
			--answer_checkpoint _0_answer_step_${step} \
			--question_checkpoint _0_question_step_${step} \
			--num_search_samples 8 \
			--batch_size 64 --gpu True \
			--ignore_unknowns False \
			--train ./zero-shot-extraction/relation_splits/train.very_small.0 \
			--dev ./zero-shot-extraction/relation_splits/dev.0 \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file $HOME/fold_1/mml-pgg-off-sim/mml-pgg-off-sim.fold.1.dev.predictions.step.${step}.csv &
	done
	wait
done
