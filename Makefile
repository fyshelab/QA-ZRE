.PHONY: install
install:

	python3 -m venv env; \
	. env/bin/activate; \
	pip3 install -e .; \
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html; \
	pip3 install nltk rouge-score sentencepiece absl-py; \
	pip3 install datasets; \
	pip3 install spacy; \

.PHONY: clean_code
clean_code:

	source env/bin/activate; \
	black src; \
	isort src; \
	docformatter --in-place src/*py

.PHONY: install_gsutil
install_gsutil:

	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-361.0.0-linux-arm.tar.gz; \
	tar -xzf google-cloud-sdk-361.0.0-linux-arm.tar.gz
	bash google-cloud-sdk/install.sh
	google-cloud-sdk/bin/gcloud init

.PHONY: train_on_lambda
train_on_lambda:

	python src/train.py --mode squad_train --model_path ./t5_squad/ learning_rate 0.001 --gpu True --gpu_device 0 --max_epochs 6 --batch_size 128

