from python:3

MAINTAINER robin.nicole.m@gmail.com

COPY . /app
WORKDIR /app

RUN pip install pipenv

RUN pipenv install --system --deploy
EXPOSE 8000
CMD ["uvicorn", "smart_annotator.api:app", "--reload", "--host", "0.0.0.0"]