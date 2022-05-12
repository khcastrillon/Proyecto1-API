from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class Model:
    def __init__(self, modelName):
        self.model = load(f"assets/{modelName}.sav")

    def make_predictions(self, data):
        print(self.model)
        vectorizer = pickle.load(open('assets/tfidf.pickle', 'rb'))
        X_val = vectorizer.transform(data)
        result = self.model.predict(X_val)
        return result

    def make_predictions_proba(self, data):
        print(self.model)
        vectorizer = pickle.load(open('assets/tfidf.pickle', 'rb'))
        X_val = vectorizer.transform(data)
        result = self.model.predict_proba(X_val)
        return result