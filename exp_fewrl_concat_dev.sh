#!/bin/bashd

seeds=(943 111 300)
steps=(1300 5000 5500)

for i in ${!seeds[@]};
do
        seed=${seeds[$i]}
        step=${steps[$i]}
        printf "step ${step} on epoch 0 for seed ${seed}\r\n"
        gsutil -m cp -r gs://emnlp-2022-rebuttal/fewrel-concat/fewrel/concat_run_${seed}/model_0_step_${step}_model ~/sep-1/fewrel/concat_run_${seed}/
        CUDA_VISIBLE_DEVICES=0 python3.7 src/re_gold_qa_train.py \
                --mode concat_fewrl_test \
                --model_path ~/sep-1/fewrel/concat_run_${seed}/ \
                --checkpoint _0_step_${step}_model \
                --learning_rate 0.0005 \
                --batch_size 128 \
                --gpu True \
                --train ./fewrl_data/train_data_${seed}.csv \
                --dev ./fewrl_data/val_data_${seed}.csv \
                --test ./fewrl_data/test_data_${seed}.csv \
                --gpu_device 0 \
                --prediction_file ~/sep-1/fewrel/concat_run_${seed}/relation.concat.run.${seed}.epoch.0.test.predictions.step.${step}.csv \
                --predict_type relation \
                --seed ${seed}
done