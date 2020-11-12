setup_repo:
	pipenv install --dev 

run-local:
	pipenv run uvicorn smart_annotator.api:app --reload

run_notebook:
	cd notebooks && pipenv run jupyter notebook

build-docker:
	docker build . -t dataset_generator/backend

run-docker: build-docker
	docker run  -p8000:8000 -t dataset_generator/backend