#!/bin/bash

'''
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
