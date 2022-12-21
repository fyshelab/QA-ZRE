# QA-ZRE
Code related to "Weakly-Supervised Questions for Zero-Shot Relation Extraction" (to appear @ EACL 2023)

# Install Requirements
Just type the following command:
```
make install
```

# Pretraining the Question Generator
This phase is slow while preprocessing the questions.
```
source env/bin/activate
mkdir -p ~/trained-models
bash pretrain_scripts/pretrain_question_module.sh MODEL_DIR=~/trained-models
```

# Pretraining the Response Generator
```
source env/bin/activate
mkdir -p ~/trained-models
bash pretrain_scripts/pretrain_response_module.sh MODEL_DIR=~/trained-models
```

# Pre-trained Weights
You can download the pre-trained weights for the question and response modules from [this link](https://drive.google.com/drive/folders/1aV03FcMrVhfwPUoc7AYY4Xg1HxKQOV8t?usp=sharing). These have been pre-trained with the batch size of 32 in a smaller (i.e. 24GB) GPU.
