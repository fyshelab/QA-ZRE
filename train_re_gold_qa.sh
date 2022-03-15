#!/bin/bash

echo -n "Enter Fold Number: " 
read fold_num

'''
#SBATCH --job-name=dev_concat_fold_1
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

'''
source env/bin/activate

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
			--mode re_classification_gold_qa_train \
		    	--model_path ~/fold_${fold_num}/gold/ \
                        --checkpoint _0_step_${step}_model \
		        --num_search_samples 8 \
		    	--batch_size 64 \
		    	--gpu True \
		    	--dev ./zero-shot-extraction/relation_splits/dev.${fold_data_id} \
		    	--gpu_device 0 \
		    	--seed 12321 \
			--prediction_file ~/fold_${fold_num}/gold/relation.gold.dev.predictions.fold.${fold_num}.step.${step}.csv \
                        --predict_type relation &
	done
	wait
done

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
