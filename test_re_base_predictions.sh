#!/bin/bash

source env/bin/activate

main_path=$HOME/t5-small-exps/naacl-2022/

'''
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path ${main_path}gold_fold_1/ \
	--checkpoint _response_pretrained \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}gold_fold_1/gold_fold_1.test.predictions.base.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path ${main_path}gold_fold_2/ \
	--checkpoint _response_pretrained \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}gold_fold_2/gold_fold_2.test.predictions.base.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path ${main_path}gold_fold_3/ \
	--checkpoint _response_pretrained \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}gold_fold_3/gold_fold_3.test.predictions.base.csv

python src/re_gold_qa_train.py \
	--mode re_concat_qa_test \
	--model_path ${main_path}concat_fold_1/ \
	--checkpoint _response_pretrained \
	--concat_questions True \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}concat_fold_1/concat_fold_1.test.predictions.base.csv

python src/re_gold_qa_train.py \
	--mode re_concat_qa_test \
	--model_path ${main_path}concat_fold_2/ \
	--checkpoint _response_pretrained \
	--concat_questions True \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}concat_fold_2/concat_fold_2.test.predictions.base.csv

python src/re_gold_qa_train.py \
	--mode re_concat_qa_test \
	--model_path ${main_path}concat_fold_3/ \
	--checkpoint _response_pretrained \
	--concat_questions True \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}concat_fold_3/concat_fold_3.test.predictions.base.csv

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ${main_path}posterier_question_model/ \
	--answer_checkpoint _response_pretrained \
	--question_checkpoint _fold_1_question_pretrained \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}posterier_question_model/base_base.test.predictions.epoch1.csv

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ${main_path}posterier_question_model/ \
	--answer_checkpoint _response_pretrained \
	--question_checkpoint _fold_2_question_pretrained \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}posterier_question_model/base_base.test.predictions.epoch2.csv
'''

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ${main_path}posterier_question_model/ \
	--answer_checkpoint _response_pretrained \
	--question_checkpoint _fold_3_question_pretrained \
	--batch_size 64  --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ${main_path}posterier_question_model/base_base.test.predictions.epoch3.csv
