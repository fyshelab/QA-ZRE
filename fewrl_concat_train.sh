#!/bin/bash

seeds=(12321 943 111 300 1300)
gpu_ids=(0 0 1 1 2)

'''
for (( e=0; e<=0; e++ ))
do
        for (( s=1; s<=13; s++ ))
        do      
                for i in ${!seeds[@]};
                do
                        step=$((s * 200))
                        seed=${seeds[$i]}
                        printf "step ${step} on epoch ${e} for seed ${seed}\r\n"
                        cuda_gpu=${gpu_ids[$i]}
                        CUDA_VISIBLE_DEVICES=${cuda_gpu} python3.7 src/re_gold_qa_train.py \
                                --mode concat_fewrl_dev \
                                --model_path ~/may-29/fewrl/concat_run_1/ \
                                --checkpoint _${e}_step_${step}_model \
                                --learning_rate 0.0005 \
                                --batch_size 128 \
                                --gpu True \
                                --train ./fewrl_data_unks/train_data_${seed}.csv \
                                --dev ./fewrl_data_unks/val_data_${seed}.csv \
                                --test ./fewrl_data_unks/test_data_${seed}.csv \
                                --gpu_device 0 \
                                --prediction_file ~/may-29/fewrl/concat_run_1/v2.relation.concat.run.${seed}.epoch.${e}.dev.predictions.step.${step}.csv \
                                --predict_type relation \
                                --seed ${seed}
                done
                wait
        done
        wait
done

for i in ${!seeds[@]};
do
        cuda_gpu=${gpu_ids[$i]}
        seed=${seeds[$i]}
        CUDA_VISIBLE_DEVICES=${cuda_gpu} python3.7 src/re_gold_qa_train.py \
                --mode multi_concat_fewrl_dev \
                --model_path ~/sep-1/fewrel/concat_run_${seed}_with_unks_more_unks/ \
                --checkpoint _response_pretrained \
                --start_epoch 0 \
                --end_epoch 0 \
                --start_step 100 \
                --end_step 26300 \
                --step_up 100 \
                --batch_size 64 \
                --gpu True \
                --train ./fewrl_data_unks_more_unks/train_data_${seed}.csv \
                --dev ./fewrl_data_unks_more_unks/val_data_${seed}.csv \
                --test ./fewrl_data_unks_more_unks/test_data_${seed}.csv \
                --gpu_device 0 \
                --predict_type relation \
                --seed ${seed} &
done
'''


for i in ${!seeds[@]};
do
        cuda_gpu=${gpu_ids[$i]}
        seed=${seeds[$i]}
        CUDA_VISIBLE_DEVICES=${cuda_gpu} python3.7 src/re_gold_qa_train.py \
                --mode concat_fewrl_train \
                --model_path ~/sep-1/fewrel/concat_run_${seed}_with_unks/ \
                --batch_size 4 \
                --max_epochs 1 \
                --checkpoint _response_pretrained \
                --learning_rate 0.0005 \
                --gpu True \
                --train ./fewrl_data_unks/train_data_${seed}.csv \
                --dev ./fewrl_data_unks/val_data_${seed}.csv \
                --test ./fewrl_data_unks/test_data_${seed}.csv \
                --gpu_device 0 \
                --seed ${seed} &
done
'''
