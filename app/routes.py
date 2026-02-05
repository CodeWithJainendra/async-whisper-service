from flask import Blueprint, request, jsonify, render_template, current_app
import tempfile
import os
from .tasks import transcribe_and_embed_task, search_audio_task  # Updated import

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/asr', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    translate_to_english = request.form.get('translate', 'false').lower() == 'true'
    
    upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    import uuid
    filename = f"{uuid.uuid4()}.webm"
    audio_path = os.path.join(upload_dir, filename)
    audio_file.save(audio_path)

    abs_audio_path = os.path.abspath(audio_path)

    # Use the new task that includes embedding
    task = transcribe_and_embed_task.delay(abs_audio_path, translate_to_english)
    
    return jsonify({
        'success': True,
        'message': 'Task submitted (Transcription + Indexing)',
        'task_id': task.id,
        'status_url': f'/tasks/{task.id}'
    }), 202

@main_bp.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')
    if not query:
        return jsonify({'error': 'Missing query parameter "q"'}), 400
        
    # Run search synchronously for simplicity (since it's fast usually)
    # In strict microservices, this might be another async task or a separate read service.
    try:
        results = search_audio_task(query)
        return jsonify({
            'success': True,
            'query': query,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    from celery.result import AsyncResult
    task_result = AsyncResult(task_id)
    
    response = {
        'task_id': task_id,
        'status': task_result.status,
    }
    
    if task_result.state == 'PENDING':
        response['result'] = None
    elif task_result.state == 'PROCESSING':
        response['result'] = task_result.info
    elif task_result.state != 'FAILURE':
        response['result'] = task_result.result
    else:
        response['error'] = str(task_result.info)
        
    return jsonify(response)
