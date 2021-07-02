#!/bin/bash
#SBATCH --job-name=reqa_train
#SBATCH --account=def-afyshe-ab
#SBATCH --nodes=2
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=0
#SBATCH --time=0-03:00
#SBATCH --cpus-per-task=6
#SBATCH --output=%N-%j.out

module load python/3.8
module load StdEnv/2020  gcc/9.3.0 arrow/2.0.0

source env/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"


echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python src/re_qa_train.py --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS --mode reqa_train --model_path $HOME/reqa_models/ --answer_checkpoint _3_model --question_checkpoint _3_model --answer_training_steps 1 --question_training_steps 3 --learning_rate 0.001 --max_epochs 4 --num_beams 16 --batch_size 8 --gpu True
