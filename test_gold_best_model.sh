#!/bin/bash

source env/bin/activate

printf "fold 1, epoch 1 best model\r\n"
gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_0_step_800_model $HOME/gold_fold_1/
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _0_step_800_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.test.predictions.0.step.800.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.test.predictions.0.step.800.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_0_step_800_model
