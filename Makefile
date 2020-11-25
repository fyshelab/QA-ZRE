.PHONY: install
install:

	python3 -m venv env; \
	. env/bin/activate; \
	pip3 install -r requirements.txt; \

.PHONY: clean_code
clean_code:

	. env/bin/activate; \
	isort src; \
	black src; \
	docformatter --in-place -r src;
