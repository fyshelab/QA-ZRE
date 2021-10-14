#!/bin/bash

source env/bin/activate

printf "fold 1, epoch 1\r\n"
for (( i=83; i<=83; i++ ))
do
	step=$((i * 100))
	printf "step ${step}\r\n"
	gsutil -m cp gs://acl-2022-storage/concat_fold_1/model_1_step_${step}_model $HOME/concat_fold_1/
	python src/re_gold_qa_train.py \
		--mode re_concat_qa_test \
		--model_path $HOME/concat_fold_1/ \
		--checkpoint _1_step_${step}_model \
		--learning_rate 0.001 --max_epochs 1 \
		--concat_questions False \
		--batch_size 64  --gpu True \
		--ignore_unknowns False \
		--train zero-shot-extraction/relation_splits/train.0 \
		--dev zero-shot-extraction/relation_splits/dev.0 \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file $HOME/concat_fold_1/concat_fold_1.dev.predictions.1.step.${step}.csv
	gsutil -m cp $HOME/concat_fold_1/concat_fold_1.dev.predictions.1.step.${step}.csv gs://acl-2022-storage/concat_fold_1/
	rm -r -f $HOME/concat_fold_1/model_1_step_${step}_model
done

'
gsutil -m cp gs://acl-2022-storage/concat_fold_1/model_1_model $HOME/concat_fold_1/
printf "Full epoch 1 \r\n"
python src/re_gold_qa_train.py \
	--mode re_concat_qa_test \
	--model_path $HOME/concat_fold_1/ \
	--checkpoint _1_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/concat_fold_1/concat_fold_1.dev.predictions.1.full.csv
gsutil -m cp $HOME/concat_fold_1/concat_fold_1.dev.predictions.1.full.csv gs://acl-2022-storage/concat_fold_1/
rm -r -f $HOME/concat_fold_1/model_1_model

gsutil -m cp gs://acl-2022-storage/concat_fold_1/model_response_pretrained_model $HOME/concat_fold_1/
printf "step 0\r\n"
python src/re_gold_qa_train.py \
	--mode re_concat_qa_test \
	--model_path $HOME/concat_fold_1/ \
	--checkpoint _response_pretrained_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/concat_fold_1/concat_fold_1.dev.predictions.1.step.0.csv
gsutil -m cp $HOME/concat_fold_1/concat_fold_1.dev.predictions.1.step.0.csv gs://acl-2022-storage/concat_fold_1/
rm -r -f $HOME/concat_fold_1/model_response_pretrained_model
'
