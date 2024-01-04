# AutoLang: Audio Language Detector

Welcome to the AutoLang repository! This powerful tool transcribes audio files and automatically detects the language of the transcribed text. It's built with Python and leverages the power of boosting trees to recognize audio and predict languages.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Command-line Arguments](#command-line-arguments)
- [Components](#components)
- [Model Training Notebooks](#model-training-notebooks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Transcribes audio files into text
- Predicts the language of the transcribed text
- Handles audio processing (changing speed and volume)
- Provides results in a user-friendly JSON format

## Quick Start

To use the program, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/EmotionEngineer/AutoLang.git
cd AutoLang
```

2. Download the model files from the [Releases](https://github.com/EmotionEngineer/AutoLang/releases/tag/v1.0.0) section of the GitHub repository. The model files have been split into four parts: `models_part_aa`, `models_part_ab`, `models_part_ac`, and `models_part_ad`.

3. Reassemble and unzip the model files:

```bash
cat models_part_aa models_part_ab models_part_ac models_part_ad > models.zip
unzip models.zip
```

4. Run the `main.py` script with the necessary command-line arguments:

```bash
python main.py -i <input_file_path> -m <model_path> -o <output_file_path> -s <speed> -v <volume> -p <processed_audio_path>
```

## Command-line Arguments

- `-i`, `--input`: Path to the input audio file. (Required)
- `-m`, `--model`: Path to the trained CatBoost LanguageDetector model. (Default: './models/LangModel.cbm')
- `-o`, `--output`: Path to the output JSON file.
- `-s`, `--speed`: Speed change of the audio file. (Default: 1.0)
- `-v`, `--volume`: Volume change of the audio file. (Default: 1.0)
- `-p`, `--path`: Path to save the processed audio file.

## Components

The program is composed of several classes:

- `Normalizer`: Normalizes audio data.
- `TextProcessor`: Processes text such as adding spaces between characters.
- `LanguageModel`: Predicts the transcription of a given audio file.
- `TextLanguagePredictor`: Predicts the language of a given text.
- `AudioLanguageDetector`: Detects the language of a given audio file.
- `AudioProcessor`: Processes the audio file with the given speed and volume.
- `JSONWriter`: Writes the result into a JSON file.

## Model Training Notebooks

The models used in this project were trained and evaluated in Kaggle notebooks. The notebooks detail the process of training the language models, inference, and conversion to ONNX format. Here are the links to those notebooks:

- [Wav2Vec2-EN-ONNX-EVAL](https://www.kaggle.com/code/tttrrraaahhh/wav2vec2-en-onnx-eval): Inference of the English version of Wav2Vec2 and conversion to ONNX format.
- [Wav2Vec2-RU-ONNX-EVAL](https://www.kaggle.com/code/tttrrraaahhh/wav2vec2-ru-onnx-eval): Inference of the Russian version of Wav2Vec2 and conversion to ONNX format.
- [RU-EN WAV2Vec with Language Auto Detection](https://www.kaggle.com/code/tttrrraaahhh/ru-en-wav2vec-with-language-auto-detection): Inference of both English and Russian models, obtaining samples, and training a CatBoost audio file language classifier based on the obtained transcriptions.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgements

This project uses the [Wav2Vec2ProcessorWithLM](https://huggingface.co/transformers/model_doc/wav2vec2.html) from HuggingFace, the [CatBoost](https://catboost.ai/) library, and [ONNX Runtime](https://onnxruntime.ai/).
