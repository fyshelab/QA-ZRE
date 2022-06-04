#!/bin/bash

source env/bin/activate

'''
CUDA_VISIBLE_DEVICES=1 python src/re_gold_qa_train.py \
    --mode fewrl_train \
    --model_path ~/fewrl/run_1/ \
    --answer_checkpoint _response_pretrained \
    --question_checkpoint _fold_1_question_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --train ./fewrl_data/train_data_12321.csv \
    --gpu_device 0 \
    --seed 12321 \
    --train_method MML-MML-Off-Sim 
'''

#run 4 test
for (( i=3; i<=3; i++ ))
do
	for (( e=0; e<=0; e++ ))
	do
		step=$(((i) * 200))
		printf "epoch ${e} & step ${step}\r\n"
		CUDA_VISIBLE_DEVICES=3 python src/re_gold_qa_train.py \
			--mode fewrl_test \
			--model_path ~/fewrl/run_4/ \
			--answer_checkpoint _${e}_answer_step_${step} \
			--question_checkpoint _${e}_question_step_${step} \
			--num_search_samples 8 \
			--batch_size 128 --gpu True \
			--test ./fewrl_data/test_data_300.csv \
			--gpu_device 0 \
			--seed 300 \
			--prediction_file ~/fewrl/run_4/relation.mml-pgg-off-sim.${e}.test.predictions.step.${step}.csv \
			--predict_type relation &
	done
	wait
done
