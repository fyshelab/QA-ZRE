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

EXPERIMENT_TYPE="qa"
TASK_NAME="response_eval"

python -m src.train \
    --batch_size 128 \
    --task_name ${TASK_NAME} \
    --t5_exp_type ${EXPERIMENT_TYPE} \
    --dev_file ./squad/dev-v2.0.json \
    --gpu True \
    --model_path ${MODEL_DIR} \
    --checkpoint "best_step" \
    --prediction_file "temp.csv" \
    --source_max_length 512 \
    --decoder_max_length 128 \
    --seed 12321
