from flask import Flask, request, jsonify, render_template
import os
import tempfile
import whisper

app = Flask(__name__)

print("Loading Whisper model...")
try:
    model = whisper.load_model("turbo")
except Exception as e:
    print(f"Error loading 'turbo' model: {e}")
    print("Falling back to 'base' model.")
    model = whisper.load_model("base")
print("Model loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/asr', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    
    # Get optional parameters from request
    translate_to_english = request.form.get('translate', 'false').lower() == 'true'
    selected_language = request.form.get('language', None)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        # User requested automatic detection based on voice
        # We will not force any language. Whisper is very good at this.
        
        args = {}
        if translate_to_english:
             args['task'] = 'translate'

        result = model.transcribe(audio_path, **args)
        
        detected_language = result.get('language', 'unknown')
        
        return jsonify({
            'success': True,
            'transcript': result['text'],
            'language': detected_language
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': 'Transcription failed', 'error': str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# Optional: Endpoint that returns both original and translated text
@app.route('/asr/full', methods=['POST'])
def transcribe_full():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        # Step 1: Transcribe in original language
        original_result = model.transcribe(audio_path)
        detected_language = original_result.get('language', 'unknown')
        
        response = {
            'success': True,
            'language': detected_language,
            'original_transcript': original_result['text'],
            'english_translation': None
        }
        
        # Step 2: If not English, also provide translation
        if detected_language != 'en':
            translated_result = model.transcribe(audio_path, task='translate')
            response['english_translation'] = translated_result['text']
        else:
            response['english_translation'] = original_result['text']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'message': 'Transcription failed', 'error': str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5011)
