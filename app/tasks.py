import whisper
import os
from . import celery

print("Loading Whisper model in worker...")
try:
    model = whisper.load_model("turbo")
    print("Turbo model loaded.")
except Exception as e:
    print(f"Error loading 'turbo' model: {e}")
    print("Falling back to 'base' model.")
    model = whisper.load_model("base")

@celery.task(bind=True)
def transcribe_audio_task(self, audio_path, translate_to_english=False):
    """
    Background task to transcribe audio.
    """
    try:
        args = {}
        if translate_to_english:
             args['task'] = 'translate'

        # Update state to processing
        self.update_state(state='PROCESSING', meta={'status': 'Transcribing...'})

        result = model.transcribe(audio_path, **args)
        
        detected_language = result.get('language', 'unknown')
        
        return {
            'success': True,
            'transcript': result['text'],
            'language': detected_language
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        # Cleanup audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

@celery.task(bind=True)
def transcribe_full_task(self, audio_path):
    """
    Background task for full transcription + translation.
    """
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Transcribing original...'})
        
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
            self.update_state(state='PROCESSING', meta={'status': 'Translating to English...'})
            translated_result = model.transcribe(audio_path, task='translate')
            response['english_translation'] = translated_result['text']
        else:
            response['english_translation'] = original_result['text']
        
        return response
        
    except Exception as e:
         return {'success': False, 'error': str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
