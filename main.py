from flask import Flask,jsonify
from transformers import pipeline
import os
app = Flask(__name__)
asr_model = pipeline('automatic-speech-recognition', model='facebook/wav2vec2-base-960h')

@app.route('/transcribe/<path:audio_path>', methods=['GET'])
def transcribe_audio(audio_path):
    # Assuming the audio file is located at the specified path
    if os.path.exists(audio_path):
        # Transcribe the audio
        transcription = asr_model(audio_path)
        return jsonify({'transcription': transcription['text']})
    else:
        return jsonify({'error': 'Audio file not found'})

if __name__ == '__main__':
    app.run(port=5000)
