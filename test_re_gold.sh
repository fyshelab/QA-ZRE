#!/bin/bash

#SBATCH --job-name=gold_test_predictions
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

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

# GOLD Fold 1
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_1/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_1/gold/base_gold_fold.1.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_1/gold/ \
	--checkpoint _0_step_800_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.0 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_1/gold/gold_fold.1.test.predictions.step.${step}.csv

# GOLD Fold 2
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_2/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_2/gold/base_gold_fold.2.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_2/gold/ \
	--checkpoint _0_step_2000_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.1 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_2/gold/gold_fold.2.test.predictions.step.${step}.csv

# GOLD Fold 3
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_3/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_3/gold/base_gold_fold.3.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_3/gold/ \
	--checkpoint _0_step_4200_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_3/gold/gold_fold.3.test.predictions.step.${step}.csv

# GOLD Fold 4
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_4/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.3 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_4/gold/base_gold_fold.4.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_4/gold/ \
	--checkpoint _0_step_1400_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.3 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_4/gold/gold_fold.4.test.predictions.step.${step}.csv

# GOLD Fold 5
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_5/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.4 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_5/gold/base_gold_fold.5.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_5/gold/ \
	--checkpoint _0_step_900_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.4 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_5/gold/gold_fold.5.test.predictions.step.${step}.csv

# GOLD Fold 6
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_6/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.5 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_6/gold/base_gold_fold.6.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_6/gold/ \
	--checkpoint _0_step_400_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.5 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_6/gold/gold_fold.6.test.predictions.step.${step}.csv

# GOLD Fold 7
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_7/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.6 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_7/gold/base_gold_fold.7.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_7/gold/ \
	--checkpoint _0_step_6100_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.6 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_7/gold/gold_fold.7.test.predictions.step.${step}.csv

# GOLD Fold 8
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_8/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.7 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_8/gold/base_gold_fold.8.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_8/gold/ \
	--checkpoint _0_step_7300_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.7 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_8/gold/gold_fold.8.test.predictions.step.${step}.csv

# GOLD Fold 9
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_9/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.8 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_9/gold/base_gold_fold.9.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_9/gold/ \
	--checkpoint _0_step_1800_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.8 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_9/gold/gold_fold.9.test.predictions.step.${step}.csv

# GOLD Fold 10
python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_10/gold/ \
	--checkpoint _response_pretrained \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.9 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_10/gold/base_gold_fold.10.test.predictions.step.${step}.csv

python src/re_gold_qa_train.py \
	--mode re_gold_qa_test \
	--model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_10/gold/ \
	--checkpoint _0_step_4100_model \
	--num_search_samples 8 \
	--batch_size 64 --gpu True \
	--ignore_unknowns False \
	--train zero-shot-extraction/relation_splits/train.0 \
	--dev zero-shot-extraction/relation_splits/test.9 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_10/gold/gold_fold.10.test.predictions.step.${step}.csv

