#!/bin/bashd

seeds=(12321 943 111 300 1300)
steps=(8100 3900 7700 8200 4200)
gpu_ids=(1 1 1 1 1)

for i in ${!seeds[@]};
do
        seed=${seeds[$i]}
        step=${steps[$i]}
        cuda_gpu=${gpu_ids[$i]}
        printf "step ${step} on epoch 0 for seed ${seed}\r\n"
        CUDA_VISIBLE_DEVICES=${cuda_gpu} python3.7 src/re_gold_qa_train.py \
                --mode concat_fewrl_test \
                --model_path ~/sep-28/wikizsl/concat_run_${seed}_with_unks/ \
                --checkpoint _0_step_${step}_model \
                --learning_rate 0.0005 \
                --batch_size 128 \
                --gpu True \
                --train ./wikizsl_data_unks/train_data_${seed}.csv \
                --dev ./wikizsl_data_unks/val_data_${seed}.csv.sampled.csv \
                --test ./wikizsl_data_unks/test_data_${seed}.csv \
                --gpu_device 0 \
                --prediction_file ~/sep-28/wikizsl/concat_run_${seed}_with_unks/relation.concat.run.${seed}.epoch.0.test.predictions.step.${step}.csv \
                --predict_type relation \
                --seed ${seed}
done
