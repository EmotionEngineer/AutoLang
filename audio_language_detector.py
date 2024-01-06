from language_model import LanguageModel
from text_language_predictor import TextLanguagePredictor

class AudioLanguageDetector:
    def __init__(self, models, text_language_predictor: TextLanguagePredictor):
        self.models = models
        self.text_language_predictor = text_language_predictor

    def detect_lang(self, file_path, manual_lang='Auto'):
        if manual_lang != 'Auto':
            temp = self.models[manual_lang].predict(file_path).text
            return temp, manual_lang
        else:
            temp = self.models['RU'].predict(file_path).text
            predicted_lang = self.text_language_predictor.predict_language(temp)
            if predicted_lang == 'EN':
                temp = self.models['EN'].predict(file_path).text
            return temp, predicted_lang
