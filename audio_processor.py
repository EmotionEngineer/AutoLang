import numpy as np
import scipy.signal as sps
from scipy.io import wavfile

class AudioProcessor:
    def __init__(self, path, speed, volume, target_path):
        self.path = path
        self.speed = speed
        self.volume = volume
        self.target_path = target_path

    def convert_to_mono(self):
        sampling_rate, data = wavfile.read(self.path)
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        return sampling_rate, data.astype(np.int16)

    def process_audio(self):
        sampling_rate, data = self.convert_to_mono()
        new_length = int(data.shape[0] / self.speed)
        new_data = sps.resample(data, new_length) if self.speed != 1.0 else data
        new_data = new_data * self.volume
        new_data = new_data.astype(data.dtype)
        wavfile.write(self.target_path, sampling_rate, new_data)
        return sampling_rate, new_data
