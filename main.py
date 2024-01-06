import argparse
from audio_processor import AudioProcessor
from audio_language_detector import AudioLanguageDetector
from json_writer import JSONWriter
from language_model import LanguageModel
from text_language_predictor import TextLanguagePredictor

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Transcribe audio files and detect language.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input audio file.')
    parser.add_argument('-m', '--model', default='./models/LangModel.cbm', help='Path to the trained CatBoost LanguageDetector model.')
    parser.add_argument('-o', '--output', help='Path to the output JSON file.')
    parser.add_argument('-s', '--speed', type=float, default=1.0, help='Speed change of the audio file.')
    parser.add_argument('-v', '--volume', type=float, default=1.0, help='Volume change of the audio file.')
    parser.add_argument('-p', '--path', type=str, default=None, help='Path to save the processed audio file.')
    
    # Add argument for manual language setting with 'Auto' as default
    parser.add_argument('-l', '--language', choices=['EN', 'RU', 'Auto'], default='Auto', help='Set the language manually (EN, RU) or leave it as Auto for auto-detection.')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()

    processed_audio_path = args.path if args.path else args.input
    audio_processor = AudioProcessor(args.input, args.speed, args.volume, processed_audio_path)
    audio_processor.process_audio()

    models = {'EN': LanguageModel('./models/wav2vec2-large-en', './models/wav2vec2-large-en/model-en.onnx'),
              'RU': LanguageModel('./models/wav2vec2-large-ru', './models/wav2vec2-large-ru/model-ru.onnx')}

    text_language_predictor = TextLanguagePredictor(args.model)
    detector = AudioLanguageDetector(models, text_language_predictor)
    transcription, detected_lang = detector.detect_lang(processed_audio_path, args.language)

    result = {"file_path": processed_audio_path, "detected_language": detected_lang, "transcription": transcription}
    json_writer = JSONWriter(result, args.output)
    json_writer.write_to_json()

if __name__ == "__main__":
    main()

