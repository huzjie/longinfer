.PHONY: install dev test lint doctor serve docker

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	python -m unittest discover -s tests -v

lint:
	python -m py_compile longinfer/*/*.py

doctor:
	longinfer doctor

serve:
	longinfer serve --port 8000

docker:
	docker build -t longinfer:latest .
