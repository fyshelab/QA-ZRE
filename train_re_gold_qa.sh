#!/bin/bash

#SBATCH --job-name=train_concat_with_unks_fold_1
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=3-00:00
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
for (( i=1; i<=1; i++ ))
do
        step=$((i * 100))
        printf "step ${step} on epoch ${i}\r\n"
        python src/re_gold_qa_train.py \
	        --mode re_gold_qa_test \
	        --model_path /home/saeednjf/scratch/fold_1/gold_with_unks/ \
                --checkpoint _response_pretrained \
		--num_search_samples 8 \
                --batch_size 64 --gpu True \
                --ignore_unknowns False \
                --train zero-shot-extraction/relation_splits/train.very_small.0 \
                --dev zero-shot-extraction/relation_splits/test.0 \
                --gpu_device 0 \
                --seed 12321 \
                --prediction_file $SCRATCH/fold_1/gold_with_unks/gold_fold.1.test.predictions.step.${step}.csv
done
'''

srun python src/re_gold_qa_train.py \
       --init_method tcp://$MASTER_ADDR:3456 \
       --world_size $SLURM_NTASKS \
       --mode re_concat_qa_train \
       --model_path /home/saeednjf/scratch/fold_1/concat_with_unks/ \
       --checkpoint _response_pretrained \
       --learning_rate 0.0005 --max_epochs 1 \
       --concat_questions True \
       --batch_size 16  --gpu True \
       --answer_training_steps 52500 \
       --ignore_unknowns True \
       --train ./zero-shot-extraction/relation_splits/train.0 \
       --dev ./zero-shot-extraction/relation_splits/dev.0 \
       --gpu_device 0 \
       --seed 12321
