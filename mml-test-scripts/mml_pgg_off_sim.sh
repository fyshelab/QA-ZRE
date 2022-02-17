#!/bin/bash

#SBATCH --job-name=train_mml_pgg_sim_off_fold_10
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=6-00:00
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
srun python src/re_gold_qa_train.py \
    --init_method tcp://$MASTER_ADDR:3456 \
    --world_size $SLURM_NTASKS \
    --mode re_qa_train \
    --model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_10/mml-pgg-off-sim/ \
    --answer_checkpoint _response_pretrained \
    --question_checkpoint _fold_1_question_pretrained \
    --training_steps 52400 \
    --learning_rate 0.0005 \
    --max_epochs 1 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --concat_questions False \
    --dev ./zero-shot-extraction/relation_splits/dev.9 \
    --train ./zero-shot-extraction/relation_splits/train.9 \
    --gpu_device 0 \
    --seed 12321 \
    --train_method MML-PGG-Off-Sim

'''

for (( i=16; i<=16; i++ ))
do
        step=$((i * 100))
        printf "step ${step} on epoch ${i}\r\n"
        python src/re_gold_qa_train.py \
                --mode re_qa_test \
		--model_path $SCRATCH/dec_29/fold_1/mml-mml-off-sim/ \
		--answer_checkpoint _0_answer_step_${step} \
                --question_checkpoint _0_question_step_${step} \
		--num_search_samples 8 \
                --batch_size 64 --gpu True \
                --ignore_unknowns True \
                --train zero-shot-extraction/relation_splits/train.very_small.0 \
                --dev zero-shot-extraction/relation_splits/test.0 \
                --gpu_device 0 \
                --seed 12321 \
                --prediction_file $SCRATCH/dec_29/fold_1/mml-mml-off-sim/mml-mml-off-sim.fold.1.test.predictions.step.${step}.csv
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