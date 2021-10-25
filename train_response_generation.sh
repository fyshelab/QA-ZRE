#!/bin/bash

source env/bin/activate

python src/response_generation/train.py --mode all_train --model_path $HOME/re_response_generation_model/ --learning_rate 0.0005 --max_epochs 6 --batch_size 16  --gpu True --gpu_device 0
