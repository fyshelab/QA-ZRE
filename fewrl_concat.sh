#!/bin/bash

'''
CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
    --mode concat_fewrl_train \
    --model_path ~/may-29/fewrl/concat_run_1/ \
    --checkpoint _response_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --train ./fewrl_data/train_data_12321.csv \
    --dev ./fewrl_data/val_data_12321.csv \
    --test ./fewrl_data/test_data_12321.csv \
    --gpu_device 0 \
    --seed 12321 \

CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
    --mode concat_fewrl_train \
    --model_path ~/may-29/fewrl/concat_run_2/ \
    --checkpoint _response_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --train ./fewrl_data/train_data_943.csv \
    --dev ./fewrl_data/val_data_943.csv \
    --test ./fewrl_data/test_data_943.csv \
    --gpu_device 0 \
    --seed 943 \


CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
    --mode concat_fewrl_train \
    --model_path ~/may-29/fewrl/concat_run_3/ \
    --checkpoint _response_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --train ./fewrl_data/train_data_111.csv \
    --dev ./fewrl_data/val_data_111.csv \
    --test ./fewrl_data/test_data_111.csv \
    --gpu_device 0 \
    --seed 111 \

CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
    --mode concat_fewrl_train \
    --model_path ~/may-29/fewrl/concat_run_4/ \
    --checkpoint _response_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --train ./fewrl_data/train_data_300.csv \
    --dev ./fewrl_data/val_data_300.csv \
    --test ./fewrl_data/test_data_300.csv \
    --gpu_device 0 \
    --seed 300 \

'''

'''
CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
    --mode concat_fewrl_train \
    --model_path ~/may-29/fewrl/concat_run_5/ \
    --checkpoint _response_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --train ./fewrl_data/train_data_1300.csv \
    --dev ./fewrl_data/val_data_1300.csv \
    --test ./fewrl_data/test_data_1300.csv \
    --gpu_device 0 \
    --seed 1300 \
'''
'''
# test for run 2.
for (( i=1; i<=1; i++ ))
do
	for (( e=0; e<=0; e++ ))
	do
		step=$((i * 200))
		printf "step ${step} on epoch ${e}\r\n"
		CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
			--mode concat_fewrl_test \
			--model_path ~/may-29/fewrl/concat_run_2/ \
			--checkpoint _${e}_step_${step}_model \
			--training_steps 2600 \
			--learning_rate 0.0005 \
			--max_epochs 1 \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--ignore_unknowns True \
			--train ./fewrl_data/train_data_943.csv \
			--dev ./fewrl_data/val_data_943.csv \
			--test ./fewrl_data/test_data_943.csv \
			--gpu_device 0 \
			--seed 943 \
			--prediction_file ~/may-29/fewrl/concat_run_2/relation.concat.run.${e}.test.predictions.step.${step}.csv \
			--predict_type relation &
	done
	wait
done

# test for run 3.
for (( i=4; i<=4; i++ ))
do
	for (( e=0; e<=0; e++ ))
	do
		step=$((i * 200))
		printf "step ${step} on epoch ${e}\r\n"
		CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
			--mode concat_fewrl_test \
			--model_path ~/may-29/fewrl/concat_run_3/ \
			--checkpoint _${e}_step_${step}_model \
			--training_steps 2600 \
			--learning_rate 0.0005 \
			--max_epochs 1 \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--ignore_unknowns True \
			--train ./fewrl_data/train_data_111.csv \
			--dev ./fewrl_data/val_data_111.csv \
			--test ./fewrl_data/test_data_111.csv \
			--gpu_device 0 \
			--seed 111 \
			--prediction_file ~/may-29/fewrl/concat_run_3/relation.concat.run.${e}.test.predictions.step.${step}.csv \
			--predict_type relation &
	done
	wait
done

# test for run 4.
for (( i=5; i<=5; i++ ))
do
	for (( e=1; e<=1; e++ ))
	do
		step=$((i * 200))
		printf "step ${step} on epoch ${e}\r\n"
		CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
			--mode concat_fewrl_test \
			--model_path ~/may-29/fewrl/concat_run_4/ \
			--checkpoint _${e}_step_${step}_model \
			--training_steps 2600 \
			--learning_rate 0.0005 \
			--max_epochs 1 \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--ignore_unknowns True \
			--train ./fewrl_data/train_data_300.csv \
			--dev ./fewrl_data/val_data_300.csv \
			--test ./fewrl_data/test_data_300.csv \
			--gpu_device 0 \
			--seed 300 \
			--prediction_file ~/may-29/fewrl/concat_run_4/relation.concat.run.${e}.test.predictions.step.${step}.csv \
			--predict_type relation &
	done
	wait
done
'''

# test for run 1.
for (( i=3; i<=3; i++ ))
do
	for (( e=0; e<=0; e++ ))
	do
		step=$((i * 200))
		printf "step ${step} on epoch ${e}\r\n"
		CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
			--mode concat_fewrl_test \
			--model_path ~/may-29/fewrl/concat_run_1/ \
			--checkpoint _${e}_step_${step}_model \
			--training_steps 2600 \
			--learning_rate 0.0005 \
			--max_epochs 1 \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--ignore_unknowns True \
			--train ./fewrl_data/train_data_12321.csv \
			--dev ./fewrl_data/val_data_12321.csv \
			--test ./fewrl_data/test_data_12321.csv \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file ~/may-29/fewrl/concat_run_1/relation.concat.run.${e}.test.predictions.step.${step}.csv \
			--predict_type relation &
	done
	wait
done
