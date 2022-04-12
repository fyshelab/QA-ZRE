#!/bin/bash

#SBATCH --job-name=train_mml_pgg_off_fold_1
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-01:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source ../dreamscape-qa/env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
python src/re_gold_qa_train.py \
    --mode re_qa_train \
    --model_path $SCRATCH/april-1/fold_1/mml-pgg-off-sim/ \
    --answer_checkpoint _response_pretrained \
    --question_checkpoint _fold_1_question_pretrained \
    --training_steps 5200 \
    --learning_rate 0.0005 \
    --max_epochs 1 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --train $SCRATCH/QA-ZRE/zero-shot-extraction/relation_splits/train.0 \
    --gpu_device 0 \
    --seed 12321 \
    --train_method MML-PGG-Off-Sim

'''
fold_num=1
for ((j=0; j<=62; j++))
do
	k=$((j * 4))
	end_k=$((k+3))
        fold_data_id=$((fold_num-1))
	for (( i=${k}; i<=${end_k}; i++ ))
	do
		step=$(((i+1) * 100))
		printf "step ${step}\r\n"
		python src/re_gold_qa_train.py \
			--mode fewrl_dev \
			--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_${fold_num}/mml-mml-off-sim/ \
			--answer_checkpoint _0_answer_step_${step} \
			--question_checkpoint _0_question_step_${step} \
			--num_search_samples 8 \
			--batch_size 16 --gpu True \
			--dev $SCRATCH/QA-ZRE/zero-shot-extraction/relation_splits/dev.${fold_data_id} \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file /home/saeednjf/scratch/feb-15-2022-arr/fold_${fold_num}/mml-mml-off-sim/relation.mml-mml-off-sim.run.0.dev.predictions.step.${step}.csv \
                        --predict_type relation &
	done
	wait
done

source ../dreamscape-qa/env/bin/activate

for (( e=0; e<=0; e++ ))
do
	for (( i=93; i<=93; i++ ))
	do
		step=$((i * 100))
		printf "step ${step} on epoch ${i}\r\n"
		python src/re_gold_qa_train.py \
			--mode fewrl_dev \
			--model_path /home/snajafi/march-23-models/fold_1/mml-pgg-off-sim/ \
			--answer_checkpoint _${e}_answer_step_${step} \
			--question_checkpoint _fold_1_question_pretrained \
			--num_search_samples 8 \
			--batch_size 8 --gpu True \
			--dev ./zero-shot-extraction/relation_splits/dev.0 \
			--gpu_device 0 \
			--seed 12321 \
			--prediction_file /home/snajafi/march-23-models/fold_1/mml-pgg-off-sim/init_q.relation.mml-pgg-off-sim.run.${e}.dev.predictions.step.${step}.csv \
			--predict_type relation
	done
done

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/fold_1/mml-pgg-off-sim/ \
	--answer_checkpoint _0_answer_full \
	--question_checkpoint _0_question_full \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/fold_1/mml-pgg-off-sim/mml_pgg_off_sim.dev.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_qa_test \

	--model_path $SCRATCH/fold_1/mml-pgg-off-sim/ \
	--answer_checkpoint _0_answer_step_500 \
	--question_checkpoint _0_question_step_500 \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/fold_1/mml-pgg-off-sim/mml_pgg_off_sim.test.predictions.step.500.csv
'''
