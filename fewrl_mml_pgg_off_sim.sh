#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
	--mode fewrl_train \
	--model_path /home/snajafi/june-19/wikizsl/run_1/ \
	--answer_checkpoint _response_pretrained_model \
	--question_checkpoint _question_second_pretrained_model \
	--training_steps 4300 \
	--learning_rate 0.0005 \
	--max_epochs 2 \
	--num_search_samples 6 \
	--batch_size 4 \
	--gpu True \
	--num_workers 3 \
	--train ~/codes/QA-ZRE/wikizsl_dataset/train_data_12321.csv \
	--dev  ~/codes/QA-ZRE/wikizsl_dataset/small.val_data_12321.csv \
	--test  ~/codes/QA-ZRE/wikizsl_dataset/test_data_12321.csv \
	--gpu_device 0 \
	--seed 12321 \
	--train_method MML-PGG-Off-Sim

'''
# test for run 5.
for (( i=12; i<=12; i++ ))
do
	for (( e=0; e<=0; e++ ))
	do
		step=$((i * 200))
		printf "step ${step} on epoch ${e}\r\n"
		CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
			--mode fewrl_test \
			--model_path ~/june-16/fewrl/run_5/ \
			--answer_checkpoint _${e}_answer_step_${step} \
			--question_checkpoint _${e}_question_step_${step} \
			--training_steps 2600 \
			--learning_rate 0.0005 \
			--max_epochs 1 \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--ignore_unknowns True \
			--train ./small_fewrl_data/train_data_1300.csv \
			--dev ./small_fewrl_data/val_data_1300.csv \
			--test ./small_fewrl_data/test_data_1300.csv \
			--gpu_device 0 \
			--seed 1300 \
			--prediction_file ~/june-16/fewrl/run_5/relation.mml-pgg-off-sim.run.${e}.test.predictions.step.${step}.csv \
			--predict_type relation
	done
done

CUDA_VISIBLE_DEVICES=1 python src/re_gold_qa_train.py \
    --mode concat_fewrl_train \
    --model_path ~/concat_run_2/ \
    --checkpoint _response_pretrained_model \
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

#SBATCH --job-name=mml-pgg-off-sim
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-04:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source ../dreamscape-qa/env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"
'''
'''
steps=(4700 400 3600 800 7900 700 2100 6800 4300 1600)
for i in ${!steps[@]};
do
	fold_num=$((i+1))
	fold_data_id=$((fold_num-1))
	step=${steps[$i]}
	CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
		--mode reqa_mml_eval \
		--model_path ~/may-20/fold_${fold_num}/ \
		--answer_checkpoint _0_answer_step_${step} \
		--question_checkpoint _0_question_step_${step} \
		--num_search_samples 8 \
		--batch_size 64 \
		--gpu True \
		--test ./zero-shot-extraction/relation_splits/test.${fold_data_id} \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file ~/may-20/fold_${fold_num}/relation.mml-pgg-off-sim.run.fold_${fold_num}.test.predictions.step.${step}.csv \
		--predict_type relation
done
'''