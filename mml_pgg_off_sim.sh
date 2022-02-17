#!/bin/bash

#SBATCH --job-name=train_mml_pgg_off_sim_fold_1
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=5-00:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

srun python src/re_gold_qa_train.py \
    --mode re_qa_train \
    --model_path $SCRATCH/feb-15-arr/fold_1/mml-pgg-off-sim/ \
    --answer_checkpoint _response_pretrained \
    --question_checkpoint _fold_1_question_pretrained \
    --training_steps 52400 \
    --learning_rate 0.0005 \
    --max_epochs 1 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --dev ./zero-shot-extraction/relation_splits/dev.0 \
    --train ./zero-shot-extraction/relation_splits/train.0 \
    --gpu_device 0 \
    --seed 12321 \
    --train_method MML-PGG-Off-Sim