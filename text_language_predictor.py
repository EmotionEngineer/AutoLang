from catboost import CatBoostClassifier
from text_processor import TextProcessor

class TextLanguagePredictor:
    def __init__(self, catboost_model):
        self.catboost_model = CatBoostClassifier().load_model(catboost_model)

    def predict_language(self, text):
        return self.catboost_model.predict([TextProcessor.add_spaces(text)])
