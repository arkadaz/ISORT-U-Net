build:
	docker-compose build
deploy:
	docker-compose up -d
all: build deploy
