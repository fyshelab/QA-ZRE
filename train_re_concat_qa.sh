#!/bin/bash

#SBATCH --job-name=reqa_train_concat_fold_1
#SBATCH --account=rrg-afyshe
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000M
#SBATCH --time=0-15:00
#SBATCH --cpus-per-task=6
#SBATCH --output=%N-%j.out

module load python/3.8
module load StdEnv/2020  gcc/9.3.0 arrow/2.0.0

source env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

python src/re_gold_qa_train.py \
       --mode re_concat_qa_train \
       --model_path $SCRATCH/t5-small-exps/concat_fold_1/ \
       --checkpoint _response_pretrained \
       --learning_rate 0.0005 --max_epochs 1 \
       --concat_questions True \
       --batch_size 64  --gpu True \
       --answer_training_steps 13125 \
       --ignore_unknowns True \
       --train ./zero-shot-extraction/relation_splits/train.0 \
       --dev ./zero-shot-extraction/relation_splits/dev.0 \
       --gpu_device 0 \
       --seed 12321
