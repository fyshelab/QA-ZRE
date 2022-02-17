#!/bin/bash

#SBATCH --job-name=reqa_gold_fold_4
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=1-00:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

source env/bin/activate

source env/bin/activate
printf "fold 1, epoch 2\r\n"
for (( i=1; i<=10; i++ ))
do
	step=$((i * 100))
	printf "step ${step}\r\n"
	python src/re_gold_qa_train.py \
		--mode re_gold_qa_test \
		--model_path $HOME/gold_fold_1/ \
		--checkpoint _0_step_${step}_model \
		--learning_rate 0.001 --max_epochs 1 \
		--concat_questions False \
		--batch_size 64  --gpu True \
		--ignore_unknowns False \
		--train zero-shot-extraction/relation_splits/train.very_small.0 \
		--dev zero-shot-extraction/relation_splits/dev.0 \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.0.step.${step}.csv
done
'''
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/mml_mml_tune/ \
	--checkpoint _response_pretrained \
	--learning_rate 0.0005 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/mml_mml_tune/gold_base.dev.predictions.0.step.0.csv
'''
'''
gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_1_model $HOME/gold_fold_1/
printf "Full epoch 1 \r\n"
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _1_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.full.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.full.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_1_model

gsutil -m cp gs://acl-2022-storage/gold_fold_1/model_response_pretrained_model $HOME/gold_fold_1/
printf "step 0\r\n"
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path $HOME/gold_fold_1/ \
	--checkpoint _response_pretrained_model \
	--learning_rate 0.001 --max_epochs 1 \
	--concat_questions False \
	--batch_size 64  --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.step.0.csv
gsutil -m cp $HOME/gold_fold_1/gold_fold_1.dev.predictions.1.step.0.csv gs://acl-2022-storage/gold_fold_1/
rm -r -f $HOME/gold_fold_1/model_response_pretrained_model
'''
