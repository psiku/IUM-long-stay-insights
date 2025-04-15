.PHONY: help install test_environment clean lint format


install:
	poetry install

test_environment:
	poetry run python test_environment.py


clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


lint:
	poetry run flake8 src


format:
	poetry run ruff format .