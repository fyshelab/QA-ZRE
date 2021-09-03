#!/bin/bash
#SBATCH --job-name=reqa_mml_pgg_top_p_iterative_train_switch_1_fold_1
#SBATCH --account=rrg-afyshe
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=0
#SBATCH --time=0-04:00
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

srun python src/re_gold_qa_train.py \
    --init_method tcp://$MASTER_ADDR:3456 \
    --world_size $SLURM_NTASKS \
    --mode re_qa_train \
    --model_path $SCRATCH/re_mml_pgg_top_p_iterative_models/switch_1/fold_1/ \
    --answer_checkpoint _response_pretrained_model \
    --question_checkpoint _question_pretrained_model \
    --training_steps 1000 \
    --update_switching_steps 1 \
    --learning_rate 0.001 \
    --max_epochs 1 \
    --num_search_samples 32 \
    --batch_size 16  \
    --gpu True \
    --num_workers 6 \
    --concat_questions false \
    --dev zero-shot-extraction/relation_splits/dev.1 \
    --train zero-shot-extraction/relation_splits/train.1
