#!/bin/bash

source env/bin/activate

main_path=$HOME/t5-small-exps/naacl-2022/

python src/re_gold_qa_train.py \
       --mode re_concat_qa_train \
       --model_path ${main_path}concat_fold_1/ \
       --checkpoint _response_pretrained \
       --learning_rate 0.0005 --max_epochs 1 \
       --concat_questions True \
       --batch_size 16  --gpu True \
       --answer_training_steps 26250 \
       --ignore_unknowns True \
       --train ./zero-shot-extraction/relation_splits/train.0 \
       --dev ./zero-shot-extraction/relation_splits/dev.0 \
       --gpu_device 0 \
       --seed 12321
