#!/bin/bash

#SBATCH --job-name=test_re_prior_type
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-05:00
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
    --model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_10/re_type_prior_lm/ \
    --checkpoint _response_pretrained \
    --training_steps 20000 \
    --learning_rate 0.0005 \
    --max_epochs 1 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --train /home/saeednjf/scratch/QA-ZRE/zero-shot-extraction/relation_splits/train.9 \
    --gpu_device 0 \
    --seed 12321 \

fold_num=10
for ((j=0; j<=65; j++))
do
	k=$((j * 4))
	end_k=$((k+3))
        fold_data_id=$((fold_num-1))
	for (( i=${k}; i<=${end_k}; i++ ))
	do
		step=$(((i+1) * 100))
		printf "step ${step}\r\n"
		python src/re_gold_qa_train.py \
			--mode relation_extraction_final_prior_test \
			--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_10/re_type_prior_lm/ \
			--checkpoint _0_step_${step}_model \
			--num_search_samples 8 \
			--batch_size 32 --gpu True \
			--dev ./zero-shot-extraction/relation_splits/dev.9 \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file /home/saeednjf/scratch/feb-15-2022-arr/fold_10/re_type_prior_lm/relation.final.re_type_prior_lm.run.0.dev.predictions.step.${step}.csv \
                        --predict_type relation &
	done
	wait
done
'''

steps=(400 300 500 300 100 2400 300 400 100 1000)
for i in ${!steps[@]};
do
	fold_num=$((i+1))
        fold_data_id=$((fold_num-1))
	step=${steps[$i]}
	python src/re_gold_qa_train.py \
		--mode relation_extraction_final_prior_test \
		--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_${fold_num}/re_type_prior_lm/ \
		--checkpoint _0_step_${step}_model \
		--num_search_samples 8 \
		--batch_size 128 --gpu True \
		--dev ./zero-shot-extraction/relation_splits/test.${fold_data_id} \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file /home/saeednjf/scratch/feb-15-2022-arr/fold_${fold_num}/re_type_prior_lm/relation.final.re_type_prior_lm.run.0.test.predictions.step.${step}.csv \
		--predict_type relation 
done
