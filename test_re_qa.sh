#!/bin/bash
source env/bin/activate
printf "fold 1, epoch 2\r\n"
for (( i=6; i<=10; i++ ))
do
	step=$((i * 100))
	printf "step ${step}\r\n"
	python src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path $HOME/mml_mml_tune/ \
		--answer_checkpoint _0_answer_step_${step} \
		--question_checkpoint _0_question_step_${step} \
		--learning_rate 0.0005 --max_epochs 1 \
		--concat_questions False \
		--batch_size 64 --gpu True \
		--ignore_unknowns False \
		--train zero-shot-extraction/relation_splits/train.very_small.0 \
		--dev zero-shot-extraction/relation_splits/dev.0 \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file $HOME/mml_mml_tune/mml_mml.dev.predictions.0.step.${step}.csv
done
'''
python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $HOME/mml_mml_tune_2/ \
	--answer_checkpoint _response_pretrained \
	--question_checkpoint _question_pretrained_second_stage \
	--learning_rate 0.0005 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/mml_mml_tune_2/base.dev.predictions.0.step.0.csv
'''
'''
gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_1_model $HOME/gold_fold_1/
printf "Full epoch 1 \r\n"
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _1_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.full.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.full.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_1_model

gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_response_pretrained_model $HOME/gold_fold_1/
printf "step 0\r\n"
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _response_pretrained_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.step.0.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.step.0.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_response_pretrained_model
'''
