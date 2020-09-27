from pydantic import BaseModel 
from typing import Union, Optional, List

class LabelMessage(BaseModel):
    success: bool = True
    message: str = 'Success'
    input: Optional[str]
    score: Optional[Union[float, None]]

class DataToLabel(BaseModel):
    id: str
    value: str