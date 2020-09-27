from typing import Optional, Union, List
from .main import DataToClassifyStd
from .models import LabelMessage, DataToLabel
from fastapi import FastAPI, Response, status
import pandas as pd 
from pathlib import Path

app = FastAPI()
DATA_TO_CLASSIFY = pd.read_csv('data/articles_to_classify.csv')
ANNOTATOR = DataToClassifyStd(DATA_TO_CLASSIFY.title.values)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/candidates", response_model=List[DataToLabel])
def read_item(response: Response):
    """
    Get the candidates data sorted the most likely to be in the category you fit for
    """
    data_list = ANNOTATOR.list_data()
    return [DataToLabel(id=data[0], value=data[1]) for data in data_list]

@app.post("/label/{item_id}")
def label(item_id: int, response: Response) -> LabelMessage:
    """
    Label on the data from the candidates data
    """
    dict_output = ANNOTATOR.label_data(item_id)
    if dict_output['X'] != dict_output['X']:
        response.status_code = 404
        return LabelMessage(success=False, message='The item id you provided does not exist')
    return LabelMessage(
        success=True,
        message='successfuly labeled',
        input=dict_output['X'], 
        score=dict_output['score'])

@app.post('/fit')
def update_scores(response: Response, candidates: Optional[bool] = True):
    """
    Refit the model on the newly labeled data
    """
    try:
        ANNOTATOR.update_score()
    except ValueError:
        return 'You need to label data before fitting'
    success_message = {'success': True, }
    if candidates:
        success_message['new_candidates'] = ANNOTATOR.list_data()
    return success_message