#!/bin/bash

source env/bin/activate

main_path=$HOME/t5-small-exps/naacl-2022/response_pretrained_model/

python src/question_response_generation/train.py \
	--mode squad_test \
	--model_path ${main_path} \
	--batch_size 8  --gpu True \
	--gpu_device 0 \
	--prediction_file ${main_path}squad_dev.epoch1.csv \
	--checkpoint _0_model

python src/question_response_generation/train.py \
	--mode squad_test \
	--model_path ${main_path} \
	--batch_size 8  --gpu True \
	--gpu_device 0 \
	--prediction_file ${main_path}squad_dev.epoch2.csv \
	--checkpoint _1_model

python src/question_response_generation/train.py \
	--mode squad_test \
	--model_path ${main_path} \
	--batch_size 8  --gpu True \
	--gpu_device 0 \
	--prediction_file ${main_path}squad_dev.epoch3.csv \
	--checkpoint _2_model

python src/question_response_generation/train.py \
	--mode squad_test \
	--model_path ${main_path} \
	--batch_size 8  --gpu True \
	--gpu_device 0 \
	--prediction_file ${main_path}squad_dev.epoch4.csv \
	--checkpoint _3_model
