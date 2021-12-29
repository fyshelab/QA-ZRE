#!/bin/bash

#SBATCH --job-name=test+reqa_pgg+mml_off_iter_fold_5
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-03:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

'''
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
python src/re_gold_qa_train.py \
	--mode re_qa_train \
	--model_path $SCRATCH/fold_5/mml-off-iter+pgg/ \
	--answer_checkpoint _response_pretrained \
	--question_checkpoint _question_best_mml_off \
	--training_steps 26250 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 16 \
	--gpu True \
	--num_workers 3 \
	--concat_questions False \
	--dev ./zero-shot-extraction/relation_splits/dev.4 \
	--train ./zero-shot-extraction/relation_splits/train.4 \
	--gpu_device 0 \
	--seed 12321 \
	--train_method Answer-PGG
'''

for (( i=97; i<=97; i++ ))
do
        step=$((i * 100))
        printf "step ${step} on epoch ${i}\r\n"
        python src/re_gold_qa_train.py \
                --mode re_qa_test \
		--model_path $SCRATCH/fold_5/mml-off-iter+pgg/ \
                --answer_checkpoint _0_answer_step_${step} \
                --question_checkpoint _0_question_step_${step} \
		--num_search_samples 8 \
                --batch_size 64 --gpu True \
                --ignore_unknowns True \
                --train zero-shot-extraction/relation_splits/train.very_small.0 \
                --dev zero-shot-extraction/relation_splits/test.4 \
                --gpu_device 0 \
                --seed 12321 \
                --prediction_file $SCRATCH/fold_5/mml-off-iter+pgg/mml_off_iter+pgg.test.predictions.step.${step}.csv
done

'''
python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/fold_1/mml-off-iter+pgg/ \
	--answer_checkpoint _0_answer_step_1300 \
	--question_checkpoint _0_question_step_1300 \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/fold_1/mml-off-iter+pgg/mml_off_iter+pgg.test.predictions.step.1300.csv
'''
