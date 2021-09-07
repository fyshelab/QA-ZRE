#!/bin/bash

source env/bin/activate
'
for (( i=0; i<=9; i++ ))
do
    python src/re_gold_qa_train.py \
       --mode re_concat_qa_train \
       --model_path $HOME/august_25_runs/re_concat_qa_models_with_unknowns/fold_$i/ \
       --checkpoint _response_pretrained_model \
       --learning_rate 0.001 --max_epochs 1 \
       --concat_questions True \
       --batch_size 16  --gpu True \
       --answer_training_steps 4000 \
       --ignore_unknowns False \
       --train zero-shot-extraction/relation_splits/train.$i \
       --dev zero-shot-extraction/relation_splits/dev.$i \
       --gpu_device 0
done
'
python src/re_gold_qa_train.py \
       --mode re_concat_qa_train \
       --model_path $HOME/august_25_runs/re_concat_qa_models_with_unknowns/fold_1/ \
       --checkpoint _response_pretrained_model \
       --learning_rate 0.001 --max_epochs 1 \
       --concat_questions True \
       --batch_size 2  --gpu True \
       --answer_training_steps 1000 \
       --ignore_unknowns False \
       --train zero-shot-extraction/relation_splits/train.1 \
       --dev zero-shot-extraction/relation_splits/dev.1 \
       --gpu_device 0
