#!/bin/bash

#SBATCH --job-name=reqa_pgg+pg_iter_off_fold_0
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-06:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
for (( i=179; i<=262; i++ ))
do
        step=$((i * 100))
        printf "step ${step} on epoch ${i}\r\n"
        python src/re_gold_qa_train.py \
                --mode re_qa_test \
		--model_path $SCRATCH/fold_1/pg-off-iter+pgg/ \
                --answer_checkpoint _0_answer_step_${step} \
                --question_checkpoint _0_question_step_${step} \
		--num_search_samples 8 \
                --batch_size 64 --gpu True \
                --ignore_unknowns True \
                --train zero-shot-extraction/relation_splits/train.very_small.0 \
                --dev zero-shot-extraction/relation_splits/dev.0 \
                --gpu_device 0 \
                --seed 12321 \
                --prediction_file $SCRATCH/fold_1/pg-off-iter+pgg/pg_off_iter+pgg.dev.predictions.step.${step}.csv
done

'''
python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/fold_1/pg-on-iter/ \
	--answer_checkpoint _0_answer_full \
	--question_checkpoint _0_question_full \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/dev.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/fold_1/pg-on-iter/pg_on_iter.dev.predictions.step.${step}.csv
'''
'''
python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/fold_1/pg-off-iter+pgg/ \
	--answer_checkpoint _0_answer_step_6400 \
	--question_checkpoint _0_question_step_6400 \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns True \
	--train zero-shot-extraction/relation_splits/train.very_small.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/fold_1/pg-off-iter+pgg/pg_off_iter+pgg.test.predictions.step.6400.csv
'''
