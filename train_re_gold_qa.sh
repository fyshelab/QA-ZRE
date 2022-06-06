#!/bin/bash


#SBATCH --job-name=dev_concat_fold_10
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=4-00:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source ../dreamscape-qa/env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

for ((j=0; j<=130; j++))
do
	k=$((j * 4))
	end_k=$((k+3))
        fold_data_id=$((fold_num-1))
	for (( i=${k}; i<=${end_k}; i++ ))
	do
		step=$(((i+1) * 100))
		printf "step ${step}\r\n"
		python src/re_gold_qa_train.py \
			--mode re_classification_qa_train \
		    	--model_path ~/fold_${fold_num}/concat/ \
                        --checkpoint _0_step_${step}_model \
		        --num_search_samples 8 \
		    	--batch_size 64 \
		    	--gpu True \
		    	--dev ./zero-shot-extraction/relation_splits/dev.${fold_data_id} \
		    	--gpu_device 0 \
		    	--seed 12321 \
			--concat True \
			--prediction_file ~/fold_${fold_num}/concat/relation.concat.dev.predictions.fold.${fold_num}.step.${step}.csv \
                        --predict_type relation &
	done
	wait
done

'''

'''
CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
		--mode re_classification_qa_train \
		--model_path ~/may-20/fold_8/concat/ \
		--checkpoint _0_step_400_model \
		--num_search_samples 8 \
		--batch_size 128 \
		--gpu True \
		--dev ./zero-shot-extraction/relation_splits/test.7 \
		--gpu_device 0 \
		--seed 12321 \
		--concat True \
		--prediction_file ~/may-20/fold_8/concat/relation.concat.test.predictions.fold.8.step.400.csv \
		--predict_type relation
CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
		--mode re_classification_qa_train \
		--model_path ~/may-20/fold_2/gold/ \
		--checkpoint _0_step_1900_model \
		--num_search_samples 8 \
		--batch_size 128 \
		--gpu True \
		--dev ./zero-shot-extraction/relation_splits/test.1 \
		--gpu_device 0 \
		--seed 12321 \
		--concat False \
		--prediction_file ~/may-20/fold_2/gold/relation.gold.test.predictions.fold.2.step.1900.csv \
		--predict_type relation

CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
		--mode re_classification_qa_train \
		--model_path ~/may-20/fold_7/gold/ \
		--checkpoint _0_step_2600_model \
		--num_search_samples 8 \
		--batch_size 128 \
		--gpu True \
		--dev ./zero-shot-extraction/relation_splits/test.6 \
		--gpu_device 0 \
		--seed 12321 \
		--concat False \
		--prediction_file ~/may-20/fold_7/gold/relation.gold.test.predictions.fold.7.step.2600.csv \
		--predict_type relation

'''
'''
#steps=(4700 10300 1400 800 14700 1100 2100 800 1300 1600)
steps=(4700)
for i in ${!steps[@]};
do
	fold_num=$((i+1))
    fold_data_id=$((fold_num-1))
	step=${steps[$i]}
	CUDA_VISIBLE_DEVICES=3 python3.7 src/re_gold_qa_train.py \
		--mode fewrl_dev \
		--model_path ~/may-20/fold_${fold_num}/ \
		--answer_checkpoint _0_answer_step_${step} \
		--question_checkpoint _0_question_step_${step} \
		--num_search_samples 8 \
		--batch_size 64 \
		--gpu True \
		--dev ./zero-shot-extraction/relation_splits/test.${fold_data_id} \
		--gpu_device 0 \
		--seed 12321 \
		--prediction_file ~/may-20/fold_${fold_num}/relation.mml-pgg-off-sim.run.fold_${fold_num}.test.predictions.step.${step}.csv \
		--predict_type relation
done
'''

#fold_num=5
#fold_data_id=4
#step=14700
#CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
#	--mode fewrl_dev \#
	--model_path ~/may-20/fold_${fold_num}/ \
	--answer_checkpoint _0_answer_step_${step} \
	--question_checkpoint _0_question_step_${step} \
	--num_search_samples 8 \
	--batch_size 64 \
	--gpu True \
	--dev ./zero-shot-extraction/relation_splits/test.${fold_data_id} \
	--gpu_device 0 \
	--seed 12321 \
	--prediction_file ~/may-20/fold_${fold_num}/relation.mml-pgg-off-sim.run.fold_${fold_num}.test.predictions.step.${step}.csv \
	--predict_type relation

'''
for (( i=1; i<=525; i++ ))
do
        step=$((i * 100))
        printf "step ${step} on epoch ${i}\r\n"
        python src/re_gold_qa_train.py \
	        --mode re_concat_qa_test \
	        --model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_1/concat/ \
                
                --batch_size 64 --gpu True \
                --ignore_unknowns False \
                --train zero-shot-extraction/relation_splits/train.0 \
                --dev zero-shot-extraction/relation_splits/dev.0 \
                --gpu_device 0 \
                --seed 12321 \
                --prediction_file $SCRATCH/feb-15-2022-arr/fold_1/concat/concat_fold.1.dev.predictions.step.${step}.csv
done

srun python src/re_gold_qa_train.py \
       --init_method tcp://$MASTER_ADDR:3456 \
       --world_size $SLURM_NTASKS \
       --mode re_gold_qa_train \
       --model_path /home/saeednjf/scratch/feb-15-2022-arr/fold_10/gold/ \
       --checkpoint _response_pretrained \
       --learning_rate 0.0005 --max_epochs 1 \
       --batch_size 16  --gpu True \
       --answer_training_steps 52400 \
       --ignore_unknowns False \
       --train ./zero-shot-extraction/relation_splits/train.9 \
       --dev ./zero-shot-extraction/relation_splits/dev.9 \
       --gpu_device 0 \
       --seed 12321
'''
