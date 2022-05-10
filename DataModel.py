from __future__ import annotations
from typing import List
from pydantic import BaseModel

class DataModel(BaseModel):
# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    life_expectancy: float
    adult_mortality: float

    #Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ["Adult Mortality", "infant deaths", "Alcohol","percentage expenditure","Hepatitis B", "Measles", "BMI",
                "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS", "DGP", "Population",
                "thinness 10-19 years", "thinness 5-9 years", "Income composition of resources", "Schooling"]
