#!/bin/bash
#SBATCH --job-name=reqa_question_generation_pretrain
#SBATCH --account=rrg-afyshe
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=0
#SBATCH --time=0-23:59
#SBATCH --cpus-per-task=6
#SBATCH --output=%N-%j.out

module load python/3.8
module load StdEnv/2020  gcc/9.3.0 arrow/2.0.0

source env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

echo "All the allocated nodes: $SLURM_JOB_NODELIST"

python src/question_response_generation/train.py --question_training True --mode all_train --model_path $SCRATCH/re_question_generation_model/ --learning_rate 0.001 --max_epochs 6 --batch_size 64 --gpu True --gpu_device 0
