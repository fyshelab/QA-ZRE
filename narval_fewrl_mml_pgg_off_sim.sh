#!/bin/bash

#SBATCH --job-name=943-offmml-g-fewrel-without-unks
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24000M
#SBATCH --time=0-05:00
#SBATCH --cpus-per-task=3
#SBATCH --output=%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0

source ../dreamscape-qa/env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

# seeds=(12321 943 111 300 1300)
# gpu_ids=(0 0 0 0 0)

seeds=(943)
gpu_ids=(0)

for i in ${!seeds[@]};
do
    cuda_gpu=${gpu_ids[$i]}
    seed=${seeds[$i]}
    CUDA_VISIBLE_DEVICES=${cuda_gpu} python src/re_gold_qa_train.py \
		--mode multi_fewrl_dev \
		--model_path ~/scratch/fewrl-runs-sep-13/fewrl-offmml-pgg/run_${seed}/ \
		--answer_checkpoint _response_pretrained \
		--question_checkpoint _fold_1_question_pretrained \
		--learning_rate 0.0005 \
		--training_steps 10600 \
		--start_epoch 0 \
		--end_epoch 0 \
		--start_step 100 \
		--end_step 10500 \
		--step_up 100 \
		--max_epochs 1 \
		--num_search_samples 8 \
		--batch_size 128 \
		--gpu True \
		--train ./fewrl_data/train_data_${seed}.csv \
		--dev  ./fewrl_data/val_data_${seed}.csv \
		--test  ./fewrl_data/test_data_${seed}.csv \
		--gpu_device 0 \
		--predict_type relation \
		--seed ${seed}
done
