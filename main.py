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
def make_predictions(dataMist: DataModel):
    logicRegresion = "LR_model"
    df = pd.DataFrame(dataMist.dict(), columns=dataMist.dict().keys(), index=[0])
    df.columns = dataMist.columns()
    # model = load("assets/pipeline.joblib")
    model = PredictionModel.Model(df.columns, logicRegresion)
    result = PredictionModel.Model.make_predictions(model, df)
    # result = model.predict(df)
    return result

@app.post("/predict2")
def make_predictions(dataList: DataList):
    df = pd.DataFrame(dataList.dict(), columns=dataList.dict().keys(), index=[0])
    df.columns = dataList.columns()
    X = df.drop('life_expectancy', axis=1)
    y = df['life_expectancy']
    # model = load("assets/pipeline.joblib")
    result = PredictionModel.Model.R2(X, y)
    # result = model.predict(df)
    return result

@app.post("/predict3")
def make_predictions(dataList: DataList):
    df = pd.DataFrame(dataList.dict(), columns=dataList.dict().keys(), index=[0])
    df.columns = dataList.columns()
    X = df.drop('life_expectancy', axis=1)
    y = df['life_expectancy']
    # model = load("assets/pipeline.joblib")
    result = PredictionModel.Model.R2(X, y)
    # result = model.predict(df)
    return result
