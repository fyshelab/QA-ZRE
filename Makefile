.PHONY: install
install:

	python3.7 -m venv env; \
	. env/bin/activate; \
	pip3 install -e .[dev]


.PHONY: clean_code
clean_code:

	source env/bin/activate; \
	black src/*py; \
	isort src/*py; \
	docformatter --in-place src/*py


