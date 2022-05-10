from joblib import load
import Procesamiento


class Model:
    def __init__(self,columns, modelName):
        self.model = load(f"assets/{modelName}.joblib")

    def make_predictions(self, data):
        print(self.model)
        result = self.model.predict(data)
        return result

