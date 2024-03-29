#!/bin/bash

'''

#SBATCH --job-name=test_fold_10
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-12:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

srun python src/re_gold_qa_train.py \
    --mode re_qa_train \
    --model_path ~/fold_4/mml-pgg-on-sim/ \
    --answer_checkpoint _response_pretrained \
    --question_checkpoint _fold_1_question_pretrained \
    --training_steps 10000 \
    --learning_rate 0.0005 \
    --max_epochs 1 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --dev ./zero-shot-extraction/relation_splits/dev.3 \
    --train ./zero-shot-extraction/relation_splits/train.3 \
    --gpu_device 0 \
    --seed 12321 \
    --train_method MML-PGG-On-Sim
'''

`

'''
python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_8/mml-pgg-off-sim/ \
	--answer_checkpoint _0_answer_step_8600 \
	--question_checkpoint _0_question_step_8600 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/dev.7 \
	--train ./zero-shot-extraction/relation_splits/train.7 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_8/mml-pgg-off-sim/mml-pgg-off-sim.dev.predictions.fold.8.step.8600.csv

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_2/mml-pgg-off-sim/ \
	--answer_checkpoint _0_answer_step_9300 \
	--question_checkpoint _0_question_step_9300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 64 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.1 \
	--train ./zero-shot-extraction/relation_splits/train.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_2/mml-pgg-off-sim/mml-pgg-off-sim.fold.2.test.predictions.step.9300.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_3/mml-pgg-off-sim/ \
	--answer_checkpoint _0_answer_step_5300 \
	--question_checkpoint _0_question_step_5300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 64 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.2 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_3/mml-pgg-off-sim/mml-pgg-off-sim.fold.3.test.predictions.step.5300.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_10/mml-pgg-off-sim/ \
	--answer_checkpoint _0_answer_step_8600 \
	--question_checkpoint _0_question_step_8600 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 64 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.9 \
	--train ./zero-shot-extraction/relation_splits/train.9 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_10/mml-pgg-off-sim/mml-pgg-off-sim.fold.10.test.predictions.step.8600.csv & 

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_2/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_9200 \
	--question_checkpoint _0_question_step_9200 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.1 \
	--train ./zero-shot-extraction/relation_splits/train.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_2/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.2.step.9200.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_2/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_9400 \
	--question_checkpoint _0_question_step_9400 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.1 \
	--train ./zero-shot-extraction/relation_splits/train.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_2/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.2.step.9400.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_2/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_9300 \
	--question_checkpoint _0_question_step_9300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.1 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_2/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.2.step.9300.csv & 

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_9/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_2300 \
	--question_checkpoint _0_question_step_2300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.8 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_9/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.9.step.2300.csv & 

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_9/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_1900 \
	--question_checkpoint _0_question_step_1900 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.8 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_9/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.9.step.1900.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_9/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_4700 \
	--question_checkpoint _0_question_step_4700 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.8 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_9/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.9.step.4700.csv &

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_8/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_17000 \
	--question_checkpoint _0_question_step_17000 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.7 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_8/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.8.step.17000.csv & 

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_8/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_7200 \
	--question_checkpoint _0_question_step_7200 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.7 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_8/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.8.step.7200.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_8/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_7900 \
	--question_checkpoint _0_question_step_7900 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.7 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_8/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.8.step.7900.csv &

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_7/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_9800 \
	--question_checkpoint _0_question_step_9800 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.6 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_7/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.7.step.9800.csv & 

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_7/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_16600 \
	--question_checkpoint _0_question_step_16600 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.6 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_7/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.7.step.16600.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_7/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_13400 \
	--question_checkpoint _0_question_step_13400 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.6 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_7/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.7.step.13400.csv &

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_6/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_300 \
	--question_checkpoint _0_question_step_300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.5 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_6/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.6.step.300.csv & 

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_6/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_2200 \
	--question_checkpoint _0_question_step_2200 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.5 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_6/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.6.step.2200.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_6/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_300 \
	--question_checkpoint _0_question_step_300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.5 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_6/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.6.step.300.csv &

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_5/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_12600 \
	--question_checkpoint _0_question_step_12600 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.4 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_5/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.5.step.12600.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_5/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_12800 \
	--question_checkpoint _0_question_step_12800 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.4 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_5/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.5.step.12800.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_5/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_3000 \
	--question_checkpoint _0_question_step_3000 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.4 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_5/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.5.step.3000.csv &

wait

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_5/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_12600 \
	--question_checkpoint _0_question_step_12600 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.4 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_5/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.5.step.12600.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_5/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_12800 \
	--question_checkpoint _0_question_step_12800 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.4 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_5/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.5.step.12800.csv &

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path ~/fold_5/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_3000 \
	--question_checkpoint _0_question_step_3000 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.4 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/fold_5/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.5.step.3000.csv &
'''
