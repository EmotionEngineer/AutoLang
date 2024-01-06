import json

class JSONWriter:
    def __init__(self, result, output):
        self.result = result
        self.output = output

    def write_to_json(self):
        if self.output:
            with open(self.output, 'w', encoding='utf8') as f:
                json.dump(self.result, f, ensure_ascii=False)
            print(f'JSON saved to: {self.output}')
            print(f'Input file path: {self.result["file_path"]}')
            print(f'Detected language: {self.result["detected_language"]}')
            print(f'Transcription: {self.result["transcription"]}')
