from typing import Optional
import pandas as pd
from joblib import load
from fastapi import FastAPI
from DataModel import DataModel
from DataList import DataList
import PredictionModel

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict1")
def make_predictions1(dataModel: DataModel):
    logicRegresion = "LR_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    # model = load("assets/pipeline.joblib")
    model = PredictionModel.Model(df.columns, logicRegresion)
    result = PredictionModel.Model.make_predictions(model, df)
    # result = model.predict(df)
    return result

@app.post("/predict2")
def make_predictions2(dataModel: DataModel):
    logicRegresion = "NB_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    # model = load("assets/pipeline.joblib")
    model = PredictionModel.Model(df.columns, logicRegresion)
    result = PredictionModel.Model.make_predictions(model, df)
    # result = model.predict(df)
    return result

@app.post("/predict3")
def make_predictions3(dataModel: DataModel):
    logicRegresion = "SVM_model"
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = PredictionModel.Model(df.columns, logicRegresion)
    result = PredictionModel.Model.make_predictions(model, df)
    return result

