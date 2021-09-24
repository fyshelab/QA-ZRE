#!/bin/bash

source env/bin/activate
'
for (( i=0; i<=9; i++ ))
do
	printf "fold ${i}"
	for (( j=1; j<=39; j++ ))
	do
		step=$((j * 100))
		printf "step ${step}"
	    	python src/re_gold_qa_train.py \
			--mode re_gold_qa_test \
	       		--model_path $HOME/august_25_runs/re_gold_qa_models_with_unknowns/fold_$i/ \
	       		--checkpoint _0_step_${step}_model \
	       		--learning_rate 0.001 --max_epochs 1 \
	       		--concat_questions False \
	       		--batch_size 64  --gpu True \
	       		--ignore_unknowns False \
	       		--train zero-shot-extraction/relation_splits/train.$i \
	       		--dev zero-shot-extraction/relation_splits/dev.$i \
	       		--gpu_device 0 \
			--prediction_file $HOME/august_25_runs/re_gold_qa_models_with_unknowns/fold_$i/dev.predictions.0.step.${step}.csv
    	done
	python src/re_gold_qa_train.py \
			--mode re_gold_qa_test \
	       		--model_path $HOME/august_25_runs/re_gold_qa_models_with_unknowns/fold_$i/ \
	       		--checkpoint _0_model \
	       		--learning_rate 0.001 --max_epochs 1 \
	       		--concat_questions False \
	       		--batch_size 64  --gpu True \
	       		--ignore_unknowns False \
	       		--train zero-shot-extraction/relation_splits/train.$i \
	       		--dev zero-shot-extraction/relation_splits/dev.$i \
	       		--gpu_device 0 \
			--prediction_file $HOME/august_25_runs/re_gold_qa_models_with_unknowns/fold_$i/dev.predictions.0.full.csv
done
'
'
python src/re_gold_qa_train.py \
			--mode re_gold_qa_test \
	       		--model_path $HOME/august_25_runs/re_gold_qa_models_with_unknowns/fold_1/ \
	       		--checkpoint _response_pretrained_model \
	       		--learning_rate 0.001 --max_epochs 1 \
	       		--concat_questions False \
	       		--batch_size 16  --gpu True \
	       		--ignore_unknowns False \
	       		--train zero-shot-extraction/relation_splits/train.1 \
	       		--dev zero-shot-extraction/relation_splits/dev.1 \
	       		--gpu_device 0 \
			--prediction_file $HOME/august_25_runs/re_gold_qa_models_with_unknowns/fold_1/dev.predictions.0.step.0.csv
'
for (( j=9; j<=25; j++ ))
	do
		step=$((j * 100))
		printf "step ${step}"
	    	python src/re_gold_qa_train.py \
			--mode re_gold_qa_test \
	       		--model_path $HOME/september_22/gold/ \
	       		--checkpoint _0_step_${step}_model \
	       		--learning_rate 0.001 --max_epochs 1 \
	       		--concat_questions False \
	       		--batch_size 16  --gpu True \
	       		--ignore_unknowns True \
	       		--train zero-shot-extraction/relation_splits/train.very_small.1 \
	       		--dev zero-shot-extraction/relation_splits/dev.1 \
	       		--gpu_device 0 \
			--prediction_file $HOME/september_22/gold/dev.predictions.0.step.${step}.csv
    	done
