#!/bin/bash
#SBATCH --job-name=reqa_train
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=0
#SBATCH --time=0-03:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --output=%N-%j.out

module load python/3.8

source ~/env/bin/activate


python src/re_qa_train.py --mode reqa_train --model_path t5_all/ --answer_checkpoint _3_model --question_checkpoint _3_model --answer_training_steps 1 --question_training_steps 3 --learning_rate 0.001 --max_epochs 4 --num_beams 16 --batch_size 4 --gpu True
