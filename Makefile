APP=app.main:app
PYTHONPATH=backend

install:
	poetry install

run:
	PYTHONPATH=$(PYTHONPATH) poetry run uvicorn $(APP) --reload --host 0.0.0.0 --port 8000

lint:
	poetry run ruff check .
	poetry run black --check .

format:
	poetry run black .
	poetry run ruff check --fix .

test:
	poetry run pytest -v

migrate:
	alembic upgrade head

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build