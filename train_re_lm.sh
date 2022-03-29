#!/bin/bash

#SBATCH --job-name=train_re_prior_lm_fold_1
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
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
python src/re_gold_qa_train.py \
    --mode relation_extraction_prior_lm_train \
    --model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_1/re_prior_lm/ \
    --checkpoint _response_pretrained \
    --training_steps 5200 \
    --learning_rate 0.0005 \
    --max_epochs 1 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --train /home/saeednjf/scratch/QA-ZRE/zero-shot-extraction/relation_splits/train.0 \
    --gpu_device 0 \
    --seed 12321 \
'''

fold_num=1
for ((j=0; j<=25; j++))
do
	k=$((j * 4))
	end_k=$((k+3))
        fold_data_id=$((fold_num-1))
	for (( i=${k}; i<=${end_k}; i++ ))
	do
		step=$(((i+1) * 100))
		printf "step ${step}\r\n"
		python src/re_gold_qa_train.py \
			--mode relation_extraction_prior_lm_test \
			--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_1/re_prior_lm/ \
			--checkpoint _0_step_${step}_model \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--dev ./zero-shot-extraction/relation_splits/dev.0 \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file /home/saeednjf/scratch/feb-15-2022-arr/fold_1/re_prior_lm/re_prior_lm.run.0.dev.predictions.step.${step}.csv \
                        --predict_type relation &
	done
	wait
done
