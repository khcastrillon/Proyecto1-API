from joblib import load

class Model:
    def __init__(self,columns, modelName):
        self.model = load(f"assets/{modelName}.sav")

    def make_predictions(self, data):
        print(self.model)
        result = self.model.predict(data)
        return result

    # def R2(self, data, y):
    #     r2 = self.model.score(data,y)
    #     return r2
