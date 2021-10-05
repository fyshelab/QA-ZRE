#!/bin/bash

source env/bin/activate

printf "fold 1, epoch 1\r\n"
for (( i=1; i<=131; i++ ))
do
	step=$((i * 100))
	printf "step ${step}\r\n"
	gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_0_step_${step}_model $HOME/gold_fold_1/
	python src/re_gold_qa_train.py \
		--mode re_gold_qa_test \
		--model_path $HOME/gold_fold_1/ \
		--checkpoint _0_step_${step}_model \
		--learning_rate 0.001 --max_epochs 1 \
		--concat_questions False \
		--batch_size 16  --gpu True \
		--ignore_unknowns False \
		--train zero-shot-extraction/relation_splits/train.0 \
		--dev zero-shot-extraction/relation_splits/dev.0 \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.step.${step}.csv
	gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.step.${step}.csv gs://acl-2022-storage/gold_fold_1/
	rm -r -f $HOME/gold_fold_1/model_0_step_${step}_model
done

gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_0_model $HOME/gold_fold_1/
printf "Full epoch 1 \r\n"
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _0_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 16  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.full.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.full.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_0_model

gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_response_pretrained_model $HOME/gold_fold_1/
printf "step 0\r\n"
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _response_pretrained_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 16  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.step.0.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.step.0.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_response_pretrained_model
