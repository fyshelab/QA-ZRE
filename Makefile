.PHONY: install
install:

	python3 -m venv env; \
	source ~/projects/def-afyshe-ab/saeednjf/codes/dreamscape-qa/env/bin/activate; \
	pip3 install -e .; \
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html; \
	pip3 install nltk rouge-score sentencepiece absl-py; \



.PHONY: clean_code
clean_code:

	source env/bin/activate; \
	black src; \
	isort src; \
	docformatter --in-place src/*py


.PHONY: train_on_lambda
train_on_lambda:

	python src/train.py --mode squad_train --model_path ./t5_squad/ learning_rate 0.001 --gpu True --gpu_device 0 --max_epochs 6 --batch_size 128

