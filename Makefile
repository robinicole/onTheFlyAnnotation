setup_repo:
	pipenv install --dev 

run_api:
	pipenv run uvicorn smart_annotator.api:app --reload

run_notebook:
	cd notebooks && pipenv run jupyter notebook

