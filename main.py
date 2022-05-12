from typing import Optional
import pandas as pd
from fastapi import FastAPI
from DataModel import DataModel
import PredictionModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins = ['*'], allow_credentials = True, allow_headers = ['*'], allow_methods = ['*'])

@app.post("/predict1")
def make_predictions1(dataModel: DataModel):
    logicRegresion = "LR_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = PredictionModel.Model(logicRegresion)
    res = PredictionModel.Model.make_predictions(model, df)
    result = {f"clasificacion: {res}, Exactitud: 81,2%"}
    return result

@app.post("/predict2")
def make_predictions2(dataModel: DataModel):
    nb = "NB_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = PredictionModel.Model( nb)
    res = PredictionModel.Model.make_predictions(model, df)
    result = {f"clasificacion: {res}, Exactitud: 80,18%"}
    return result

@app.post("/predict3")
def make_predictions3(dataModel: DataModel):
    svm = "SVM_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = PredictionModel.Model(svm)
    res = PredictionModel.Model.make_predictions(model, df)
    result = {f"clasificacion: {res}, Exactitud: 83,57%"}
    return result

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}