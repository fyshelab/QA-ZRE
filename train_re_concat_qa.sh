#!/bin/bash

#SBATCH --job-name=train_concat_fold_1
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=4-00:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

srun python src/re_gold_qa_train.py \
       --mode re_concat_qa_train \
       --model_path $SCRATCH/feb-15-arr/fold_1/concat/ \
       --checkpoint _response_pretrained \
       --learning_rate 0.0005 --max_epochs 1 \
       --batch_size 16  --gpu True \
       --train ./zero-shot-extraction/relation_splits/train.0 \
       --dev ./zero-shot-extraction/relation_splits/dev.0 \
       --gpu_device 0 \
       --seed 12321
