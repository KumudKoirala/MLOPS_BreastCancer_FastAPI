import joblib
from fastapi import FastAPI
from sklearn.datasets import load_breast_cancer
from pydantic import BaseModel, conlist
from typing import Optional,List
import pandas as pd
import uvicorn 

app=FastAPI()


class breast_cancer(BaseModel):
    data: List[conlist(float,min_items=30)]

# @app.on_event('startup')
# async def load_model():
model =joblib.load('breast_cancer_classifier.joblib')


@app.post('/predict',tags=["predictions"])
async def get_predictions(cancer:breast_cancer):
    data=dict(cancer)['data']
    prediction= model.predict(data).tolist()
    class_probability=model.predict_proba(data).tolist()
    return {"prediction":prediction,"class_probability":class_probability}



