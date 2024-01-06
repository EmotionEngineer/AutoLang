import onnxruntime as rt
from scipy.io import wavfile
import scipy.signal as sps
import numpy as np
from normalizer import Normalizer
from transformers import Wav2Vec2ProcessorWithLM

class ILanguageModel:
    def predict(self, file):
        pass

class LanguageModel(ILanguageModel):
    def __init__(self, pretrained_model, onnx_path):
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(pretrained_model)
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = rt.InferenceSession(onnx_path, sess_options)

    def predict(self, file):
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
