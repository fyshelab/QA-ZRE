.PHONY: clean_code
clean_code:

	source env/bin/activate; \
	black src/*py; \
	isort src/*py; \
	docformatter --in-place src/*py ; \

.PHONY: install_gsutil
install_gsutil:

	curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-404.0.0-linux-x86_64.tar.gz; \
	tar -xzf google-cloud-cli-404.0.0-linux-x86_64.tar.gz;
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
	pip3 install ujson; \
	pip3 install tensorboard; \
	python -m spacy download en
