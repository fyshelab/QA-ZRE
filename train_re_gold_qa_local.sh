#!/bin/bash

source env/bin/activate

python src/re_gold_qa_train.py --mode re_gold_qa_train --model_path $HOME/re_gold_qa_models_answer_only/ --checkpoint _3_model --learning_rate 0.001 --max_epochs 2 --batch_size 3  --gpu True --train zero-shot-extraction/relation_splits/train.very_small.0 --dev zero-shot-extraction/relation_splits/dev.0 --gpu_device 0
