import argparse
import json
import numpy as np
import onnxruntime as rt
import scipy.signal as sps
from abc import ABC, abstractmethod
from catboost import CatBoostClassifier
from scipy.io import wavfile
from transformers import Wav2Vec2ProcessorWithLM

class Normalizer:
    @staticmethod
    def normalize(x):
        """Normalize the input numpy audio array"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return np.squeeze((x - mean) / np.sqrt(var + 1e-5))

class TextProcessor:
    @staticmethod
    def add_spaces(text):
        """Add spaces between characters in the text"""
        return ' '.join(list(text))

class ILanguageModel(ABC):
    @abstractmethod
    def predict(self, file):
        pass

class LanguageModel(ILanguageModel):
    def __init__(self, pretrained_model, onnx_path):
        """Initialize the language model with pretrained model and ONNX path"""
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(pretrained_model)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = rt.InferenceSession(onnx_path, sess_options)

    def predict(self, file):
        """Predict the transcription of the given audio file"""
        try:
            sampling_rate, data = wavfile.read(file)
            samples = round(len(data) * float(16_000) / sampling_rate)
            new_data = sps.resample(data, samples)
            speech_array = np.array(new_data, dtype=np.float32)
            speech_array = Normalizer.normalize(speech_array)[None]
            features = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
            input_values = features.input_values
            onnx_outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_values.numpy()})[0]
            return self.processor.decode(onnx_outputs.squeeze())
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

class TextLanguagePredictor:
    def __init__(self, catboost_model):
        """Initialize the text language predictor with CatBoost model"""
        self.catboost_model = CatBoostClassifier().load_model(catboost_model)

    def predict_language(self, text):
        """Predict the language of the given text"""
        return self.catboost_model.predict([TextProcessor.add_spaces(text)])

class AudioLanguageDetector:
    def __init__(self, models, text_language_predictor: TextLanguagePredictor):
        """Initialize the audio language detector with models and text language predictor"""
        self.models = models
        self.text_language_predictor = text_language_predictor

    def auto_detect_lang(self, file_path):
        """Automatically detect the language of the given audio file"""
        temp = self.models['RU'].predict(file_path).text
        predicted_lang = self.text_language_predictor.predict_language(temp)
        if predicted_lang == 'EN':
            temp = self.models['EN'].predict(file_path).text
        return temp, predicted_lang

class AudioProcessor:
    def __init__(self, path, speed, volume):
        """Initialize the audio processor with path, speed, and volume"""
        self.path = path
        self.speed = speed
        self.volume = volume

    def process_audio(self):
        """Process the audio file with the given speed and volume"""
        sampling_rate, data = wavfile.read(self.path)
        new_length = int(data.shape[0] / self.speed)
        new_data = sps.resample(data, new_length) if self.speed != 1.0 else data
        new_data = new_data * self.volume
        new_data = new_data.astype(data.dtype)
        return sampling_rate, new_data

class JSONWriter:
    def __init__(self, result, output):
        """Initialize the JSON writer with result and output"""
        self.result = result
        self.output = output

    def write_to_json(self):
        """Write the result into a JSON file"""
        if self.output:
            with open(self.output, 'w', encoding='utf8') as f:
                json.dump(self.result, f, ensure_ascii=False)
            print(f'JSON saved to: {self.output}')
            print(f'Input file path: {self.result["file_path"]}')
            print(f'Detected language: {self.result["detected_language"]}')
            print(f'Transcription: {self.result["transcription"]}')

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Transcribe audio files and detect language.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input audio file.')
    parser.add_argument('-m', '--model', default='./models/LangModel.cbm', help='Path to the trained CatBoost LanguageDetector model.')
    parser.add_argument('-o', '--output', help='Path to the output JSON file.')
    parser.add_argument('-s', '--speed', type=float, default=1.0, help='Speed change of the audio file.')
    parser.add_argument('-v', '--volume', type=float, default=1.0, help='Volume change of the audio file.')
    parser.add_argument('-p', '--path', type=str, default=None, help='Path to save the processed audio file.')
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()

    audio_processor = AudioProcessor(args.input, args.speed, args.volume)
    sampling_rate, new_data = audio_processor.process_audio()

    processed_audio_path = args.path if args.path else args.input
    wavfile.write(processed_audio_path, sampling_rate, new_data)

    models = {'EN': LanguageModel('./models/wav2vec2-large-en', './models/wav2vec2-large-en/model-en.onnx'),
              'RU': LanguageModel('./models/wav2vec2-large-ru', './models/wav2vec2-large-ru/model-ru.onnx')}

    text_language_predictor = TextLanguagePredictor(args.model)
    detector = AudioLanguageDetector(models, text_language_predictor)
    transcription, detected_lang = detector.auto_detect_lang(processed_audio_path)

    result = {"file_path": processed_audio_path, "detected_language": detected_lang, "transcription": transcription}
    json_writer = JSONWriter(result, args.output)
    json_writer.write_to_json()

if __name__ == "__main__":
    main()

