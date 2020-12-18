.PHONY: install
install:

	python3.7 -m venv env; \
	. env/bin/activate; \
	python3.7 -m pip install -r requirements.txt; \

.PHONY: clean_code
clean_code:

	. env/bin/activate; \
	isort src; \
	black src; \
	docformatter --in-place -r src;
