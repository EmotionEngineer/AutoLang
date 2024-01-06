import numpy as np

class Normalizer:
    @staticmethod
    def normalize(x):
        """Normalize the input numpy audio array"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return np.squeeze((x - mean) / np.sqrt(var + 1e-5))
