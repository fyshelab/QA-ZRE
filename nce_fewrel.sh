#!/bin/bash

source env/bin/activate

'''
#SBATCH --job-name=test_base_fewrl_run_3
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-01:00
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

'''

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
python src/re_gold_qa_train.py \
    --mode nce_fewrl_train \
    --model_path ~/fewrl/run_1/ \
    --answer_checkpoint _3_answer_step_2600 \
    --question_checkpoint _3_question_step_2600 \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 64 \
    --gpu True \
    --num_workers 3 \
    --train ./nce_fewrel_data/nce_train_data_12321.csv \
    --gpu_device 0 \
    --seed 12321 \
    --train_method InfoNCE \
    --num_neg_samples 3 \
    --predict_type relation

'''

python src/re_gold_qa_train.py \
    --mode nce_fewrl_dev \
    --model_path ~/fewrl/run_1/ \
    --answer_checkpoint _3_answer_step_2600 \
    --question_checkpoint _3_question_step_2600 \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 64 \
    --gpu True \
    --num_workers 3 \
    --train ./nce_fewrel_data/nce_train_data_12321.csv \
    --dev ./nce_fewrel_data/val_data_12321.csv \
    --gpu_device 0 \
    --seed 12321 \
    --train_method InfoNCE \
    --num_neg_samples 3 \
    --predict_type relation \
    --prediction_file ~/fewrl/run_1/infonce.mml-pgg-off.run.3.dev.predictions.step.2600.csv
'''
for (( e=0; e<=3; e++ ))
do
	for (( i=1; i<=46; i++ ))
	do
		step=$((i * 100))
		printf "step ${step} on epoch ${i}\r\n"
		python src/re_gold_qa_train.py \
			--mode fewrl_dev \
			--model_path $HOME/wikizsl/run_5/ \
			--answer_checkpoint _${e}_answer_step_${step} \
			--question_checkpoint _${e}_question_step_${step} \
			--num_search_samples 8 \
			--training_steps 4682 \
			--batch_size 64 --gpu True \
			--train ./wikizsl_data/train_data_1.csv \
			--dev ./wikizsl_data/val_data_1.csv \
			--test ./wikizsl_data/test_data_1.csv \
			--gpu_device 0 \
			--seed 1 \
			--prediction_file $HOME/wikizsl/run_5/mml-pgg-off-sim.run.${e}.dev.predictions.step.${step}.csv
	done
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
