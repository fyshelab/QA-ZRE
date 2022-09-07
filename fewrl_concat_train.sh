#!/bin/bash

seeds=(12321 943 111 300 1300 545 12 10001 77 1993)
gpu_ids=(0 0 0 1 1 1 2 2 3 3)
for (( e=0; e<=4; e++ ))
do
        for (( s=1; s<=105; s++ ))
        do      
                for i in ${!seeds[@]};
                do
                        step=$((s * 100))
                        seed=${seeds[$i]}
                        printf "step ${step} on epoch ${e} for seed ${seed}\r\n"
                        cuda_gpu=${gpu_ids[$i]}
                        CUDA_VISIBLE_DEVICES=${cuda_gpu} python3.7 src/re_gold_qa_train.py \
                                --mode concat_fewrl_dev \
                                --model_path ~/sep-1/fewrel/concat_run_${seed}/ \
                                --checkpoint _${e}_step_${step}_model \
                                --learning_rate 0.0005 \
                                --batch_size 16 \
                                --gpu True \
                                --train ./fewrl_data/train_data_${seed}.csv \
                                --dev ./fewrl_data/val_data_${seed}.csv \
                                --test ./fewrl_data/test_data_${seed}.csv \
                                --gpu_device 0 \
                                --prediction_file ~/sep-1/fewrel/concat_run_${seed}/relation.concat.run.${seed}.epoch.${e}.dev.predictions.step.${step}.csv \
                                --predict_type relation \
                                --seed ${seed} &
                done
                wait
        done
        wait
done
