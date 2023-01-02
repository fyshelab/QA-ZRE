#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

# prepare environment
source ${VIRTUAL_ENV}/bin/activate
echo "Using Python from: $(which python)"

MODEL_PATH=${MODEL_DIR}
EXPERIMENT_TYPE="qa"
TASK_NAME="question_pretrain"

model_path=${MODEL_PATH}/${EXPERIMENT_TYPE}_${TASK_NAME}
mkdir -p ${model_path}

# training_steps is a very large value to go over everything on the train dataset.
python -m src.train \
    --batch_size 32 \
    --task_name ${TASK_NAME} \
    --t5_exp_type ${EXPERIMENT_TYPE} \
    --gpu True \
    --model_path ${model_path} \
    --learning_rate 0.0005 \
    --max_epochs 4 \
    --training_steps 10000000 \
    --source_max_length 512 \
    --decoder_max_length 128 \
    --seed 12321
