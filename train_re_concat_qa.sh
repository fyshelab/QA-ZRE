#!/bin/bash

source env/bin/activate

python src/re_gold_qa_train.py --mode re_concat_qa_train --model_path $HOME/re_concat_qa_models/ --checkpoint _3_model --learning_rate 0.001 --max_epochs 4 --batch_size 64  --gpu True --train zero-shot-extraction/relation_splits/train.0 --dev zero-shot-extraction/relation_splits/dev.0 
