pip-compile: requirements.in test-requirements.in docs-requirements.in dev-requirements.in rendering-requirements.in
	uv pip compile --no-emit-index-url --no-emit-find-links requirements.in -o requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links test-requirements.in -o test-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links docs-requirements.in -o docs-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links dev-requirements.in -o dev-requirements.txt
	uv pip compile --no-emit-index-url --no-emit-find-links rendering-requirements.in -o rendering-requirements.txt

pip-install:
	uv pip install -r dev-requirements.txt -e .

format:
	ruff format src tests
	ruff check --fix

test-coverage:
	pytest --cov-report=html --cov=src tests
