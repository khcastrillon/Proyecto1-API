from typing import Optional
import pandas as pd
from fastapi import FastAPI
from DataModel import DataModel
import PredictionModel
from fastapi.middleware.cors import CORSMiddleware
import csv

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins = ['*'], allow_credentials = True, allow_headers = ['*'], allow_methods = ['*'])

@app.post("/predict1")
def make_predictions1(dataModel: DataModel):
    logicRegresion = "LR_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df = df["study_and_condition"]
    df.columns = dataModel.columns()
    model = PredictionModel.Model(logicRegresion)
    res = PredictionModel.Model.make_predictions(model, df)
    pred = PredictionModel.Model.make_predictions_proba(model, df)
    if (res[0] == 1):
        res = "Sí"
        pred = pred[0][1]*100
    else:
        res = "No"
        pred = pred[0][0]*100
    result = {f"clasificacion: {res}, Exactitud: {pred}%"}
    return result

@app.post("/predict2")
def make_predictions2(dataModel: DataModel):
    nb = "NB_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df = df["study_and_condition"]
    df.columns = dataModel.columns()
    model = PredictionModel.Model(nb)
    res = PredictionModel.Model.make_predictions(model, df)
    pred = PredictionModel.Model.make_predictions_proba(model, df)
    if (res[0] == 1):
        res = "Sí"
        pred = pred[0][1] * 100
    else:
        res = "No"
        pred = pred[0][0] * 100
    result = {f"clasificacion: {res}, Exactitud: {pred}%"}
    return result

@app.post("/predict3")
def make_predictions3(dataModel: DataModel):
    svm = "SVM_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df = df["study_and_condition"]
    df.columns = dataModel.columns()
    model = PredictionModel.Model(svm)
    res = PredictionModel.Model.make_predictions(model, df)
    prob = 0
    if (res == [1]):
        res = "Sí"
        prob = "81,2%"
    else:
        res = "No"
        prob =  "80,06%"
    result = {f"clasificacion: {res}, Exactitud: {prob}"}
    return result

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}