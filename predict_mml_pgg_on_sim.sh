#!/bin/bash

#SBATCH --job-name=test_fold_3
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source env/bin/activate

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

'''
srun python src/re_gold_qa_train.py \
    --mode re_qa_train \
    --model_path $SCRATCH/feb-15-2022-arr/fold_4/mml-pgg-on-sim/ \
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

'''
for (( j=0; j<=24; j++))
do
	k=$((j * 4))
	end_k=$((k+3))
	for (( i=${k}; i<=${end_k}; i++ ))
	do
		step=$(((i+1) * 100))
		printf "step ${step}\r\n"
		python src/re_gold_qa_train.py \
			--mode re_qa_test \
		    	--model_path $SCRATCH/feb-15-2022-arr/fold_2/mml-pgg-on-sim/ \
		    	--answer_checkpoint _0_answer_step_${step} \
		    	--question_checkpoint _0_question_step_${step} \
		    	--training_steps 10000 \
		    	--learning_rate 0.0005 \
		    	--max_epochs 1 \
		    	--num_search_samples 8 \
		    	--batch_size 32 \
		    	--gpu True \
		    	--dev ./zero-shot-extraction/relation_splits/dev.1 \
		    	--train ./zero-shot-extraction/relation_splits/train.1 \
		    	--gpu_device 0 \
		    	--seed 12321 \
			--prediction_file $SCRATCH/feb-15-2022-arr/fold_2/mml-pgg-on-sim/mml-pgg-on-sim.dev.predictions.fold.2.step.${step}.csv &
	done
	wait
done
'''


python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/feb-15-2022-arr/fold_3/mml-pgg-on-sim/ \
	--answer_checkpoint _0_answer_step_300 \
	--question_checkpoint _0_question_step_300 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.2 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_3/mml-pgg-on-sim/mml-pgg-on-sim.test.predictions.fold.3.step.300.csv 

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/feb-15-2022-arr/fold_3/mml-mml-on-sim/ \
	--answer_checkpoint _0_answer_step_2000 \
	--question_checkpoint _0_question_step_2000 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.2 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_3/mml-mml-on-sim/mml-mml-on-sim.test.predictions.fold.3.step.2000.csv 

python src/re_gold_qa_train.py \
	--mode re_qa_test \
	--model_path $SCRATCH/feb-15-2022-arr/fold_3/mml-mml-off-sim/ \
	--answer_checkpoint _0_answer_step_3800 \
	--question_checkpoint _0_question_step_3800 \
	--training_steps 10000 \
	--learning_rate 0.0005 \
	--max_epochs 1 \
	--num_search_samples 8 \
	--batch_size 32 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.2 \
	--train ./zero-shot-extraction/relation_splits/train.2 \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file $SCRATCH/feb-15-2022-arr/fold_3/mml-mml-off-sim/mml-mml-off-sim.test.predictions.fold.3.step.3800.csv 
