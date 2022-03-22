.PHONY: install_narval
install_narval:

	. ../dreamscape-qa/env/bin/activate; \
	pip3 install -e .; \
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -U -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir; \
	pip3 install nltk rouge-score sentencepiece absl-py; \
	module load StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/5.0.0; \
	pip3 install pyarrow==5.0.0; \
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

.PHONY: install
install:

	python3 -m venv env; \
	. env/bin/activate; \
	pip3 install -e .; \
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -U -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir; \
	pip3 install nltk rouge-score sentencepiece absl-py; \
	pip3 install datasets; \
	pip3 install spacy; \
