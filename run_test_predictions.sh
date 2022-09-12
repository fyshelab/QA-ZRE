#!/bin/bash

'''
mml_mml_off=(500 9300 3800 2300 3000 300 13400 7900 4700 1500)
mml_mml_on=(8900 9400 2000 4200 12800 2200 16600 7200 1900 5800)
mml_pgg_off=(500 9300 5300 1100 6400 700 14900 3800 2900 8600)
mml_pgg_on=(300 9200 300 500 12600 300 9800 17000 2300 4300)

for i in ${!mml_mml_off[@]};
do
	fold_step=${mml_mml_off[$i]}
	fold_i=$((i+1))
	gsutil -m cp -r gs://emnlp-2022/fold_${fold_i}/mml-mml-off-sim/model*_${fold_step} ~/reqa-predictions/fold_${fold_i}/mml-mml-off-sim/
	CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path ~/reqa-predictions/fold_${fold_i}/mml-mml-off-sim/ \
		--answer_checkpoint _0_answer_step_${fold_step} \
		--question_checkpoint _0_question_step_${fold_step} \
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
		--prediction_file ~/reqa-predictions/fold_${fold_i}/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.${fold_i}.step.${fold_step}.csv
done


for i in ${!mml_mml_on[@]};
do
	fold_step=${mml_mml_on[$i]}
	fold_i=$((i+1))
	gsutil -m cp -r gs://emnlp-2022/fold_${fold_i}/mml-mml-on-sim/model*_${fold_step} ~/reqa-predictions/fold_${fold_i}/mml-mml-on-sim/
	CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path ~/reqa-predictions/fold_${fold_i}/mml-mml-on-sim/ \
		--answer_checkpoint _0_answer_step_${fold_step} \
		--question_checkpoint _0_question_step_${fold_step} \
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
		--prediction_file ~/reqa-predictions/fold_${fold_i}/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.${fold_i}.step.${fold_step}.csv
done


for i in ${!mml_pgg_off[@]};
do
	fold_step=${mml_pgg_off[$i]}
	fold_i=$((i+1))
	gsutil -m cp -r gs://emnlp-2022/fold_${fold_i}/mml-pgg-off-sim/model*_${fold_step} ~/reqa-predictions/fold_${fold_i}/mml-pgg-off-sim/
	CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path ~/reqa-predictions/fold_${fold_i}/mml-pgg-off-sim/ \
		--answer_checkpoint _0_answer_step_${fold_step} \
		--question_checkpoint _0_question_step_${fold_step} \
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
		--prediction_file ~/reqa-predictions/fold_${fold_i}/mml-pgg-off-sim/mml-pgg-off-sim.test.predictions.fold.${fold_i}.step.${fold_step}.csv
done


for i in ${!mml_pgg_on[@]};
do
	fold_step=${mml_pgg_on[$i]}
	fold_i=$((i+1))
	gsutil -m cp -r gs://emnlp-2022/fold_${fold_i}/mml-pgg-on-sim/model*_${fold_step} ~/reqa-predictions/fold_${fold_i}/mml-pgg-on-sim/
	CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
		--mode re_qa_test \
		--model_path ~/reqa-predictions/fold_${fold_i}/mml-pgg-on-sim/ \
		--answer_checkpoint _0_answer_step_${fold_step} \
		--question_checkpoint _0_question_step_${fold_step} \
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
		--prediction_file ~/reqa-predictions/fold_${fold_i}/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.${fold_i}.step.${fold_step}.csv
done
'''

mml_pgg_off=(4700 400 3600 800 7900 700 2100 6800 4300 1600)


for i in ${!mml_pgg_off[@]};
do
	fold_step=${mml_pgg_off[$i]}
	fold_i=$((i+1))
	gsutil -m cp -r gs://emnlp-2022/fold_${fold_i}/mml-pgg-off-sim/model*_${fold_step} ~/reqa-predictions/fold_${fold_i}/mml-pgg-off-sim/
	CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
		--mode reqa_mml_eval \
		--model_path ~/reqa-predictions/fold_${fold_i}/mml-pgg-off-sim/ \
		--answer_checkpoint _0_answer_step_${fold_step} \
		--question_checkpoint _0_question_step_${fold_step} \
		--training_steps 10000 \
		--learning_rate 0.0005 \
		--max_epochs 1 \
		--num_search_samples 8 \
		--batch_size 128 \
		--gpu True \
		--test ./zero-shot-extraction/relation_splits/test.${i} \
		--train ./zero-shot-extraction/relation_splits/train.${i} \
		--gpu_device 0 \
		--seed 12321 \
		--predict_type relation \
		--prediction_file ~/reqa-predictions/fold_${fold_i}/mml-pgg-off-sim/v2.relation.mml-pgg-off-sim.run.fold_${fold_i}.test.predictions.step.${predict_step}.csv
done