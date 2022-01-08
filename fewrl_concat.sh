#!/bin/bash

#SBATCH --job-name=test_concat_fewrl_run_5
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=00-01:00
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
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
srun python src/re_gold_qa_train.py \
    --init_method tcp://$MASTER_ADDR:3456 \
    --world_size $SLURM_NTASKS \
    --mode concat_fewrl_train \
    --model_path $SCRATCH/fewrl/concat_run_5/ \
    --checkpoint _response_pretrained \
    --training_steps 2600 \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --num_search_samples 8 \
    --batch_size 16 \
    --gpu True \
    --num_workers 3 \
    --train ./fewrl_data/train_data_1300.csv \
    --dev ./fewrl_data/val_data_1300.csv \
    --test ./fewrl_data/test_data_1300.csv \
    --gpu_device 0 \
    --seed 1300 \
'''

for (( e=3; e<=3; e++ ))
do
	for (( i=17; i<=17; i++ ))
	do
		step=$((i * 100))
		printf "step ${step} on epoch ${i}\r\n"
		python src/re_gold_qa_train.py \
			--mode concat_fewrl_test \
			--model_path $SCRATCH/fewrl/concat_run_5/ \
			--checkpoint _${e}_step_${step}_model \
			--training_steps 2600 \
			--learning_rate 0.0005 \
			--max_epochs 1 \
			--num_search_samples 8 \
			--batch_size 64 --gpu True \
			--ignore_unknowns True \
			--train ./fewrl_data/train_data_1300.csv \
			--dev ./fewrl_data/val_data_1300.csv \
			--test ./fewrl_data/test_data_1300.csv \
			--gpu_device 0 \
			--seed 1300 \
			--prediction_file $SCRATCH/fewrl/concat_run_5/concat.run.${e}.test.predictions.step.${step}.csv
	done
done
'''

for (( i=26; i<=26; i++ ))
do
        step=$((i * 100))
        printf "step ${step} on epoch ${i}\r\n"
        python src/re_gold_qa_train.py \
                --mode concat_fewrl_test \
		--model_path $SCRATCH/fewrl/concat_run_1/ \
                --checkpoint _0_step_${step}_model \
		--training_steps 2600 \
		--learning_rate 0.0005 \
		--max_epochs 1 \
		--num_search_samples 8 \
                --batch_size 64 --gpu True \
                --ignore_unknowns True \
		--train ./fewrl_data/train_ref_12321.csv \
	        --dev ./fewrl_data/dev_ref_12321.csv \
	        --test ./fewrl_data/train_ref_12321.csv \
                --gpu_device 0 \
                --seed 12321 \
                --prediction_file $SCRATCH/fewrl/concat_run_1/concat.run.1.train.predictions.step.${step}.csv
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
