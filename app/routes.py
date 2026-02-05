from flask import Blueprint, request, jsonify, render_template, current_app
import tempfile
import os
from .tasks import transcribe_audio_task, transcribe_full_task

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/asr', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    
    # Get optional parameters from request
    translate_to_english = request.form.get('translate', 'false').lower() == 'true'
    
    # Save to a temporary file that persists so the worker can access it
    # Note: In production with distributed workers, use shared storage (S3/NFS)
    # properly configured. For local/docker-compose, volume mount works.
    
    # We must assume the worker can see the file. 
    # Using a relative path in 'uploads' might be safer if we map it.
    
    upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # use a unique filename
    import uuid
    filename = f"{uuid.uuid4()}.webm"
    audio_path = os.path.join(upload_dir, filename)
    audio_file.save(audio_path)

    # Convert to absolute path for the worker
    abs_audio_path = os.path.abspath(audio_path)

    task = transcribe_audio_task.delay(abs_audio_path, translate_to_english)
    
    return jsonify({
        'success': True,
        'message': 'Task submitted',
        'task_id': task.id,
        'status_url': f'/tasks/{task.id}'
    }), 202

@main_bp.route('/asr/full', methods=['POST'])
def transcribe_full():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    
    upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    import uuid
    filename = f"{uuid.uuid4()}.webm"
    audio_path = os.path.join(upload_dir, filename)
    audio_file.save(audio_path)
    
    abs_audio_path = os.path.abspath(audio_path)

    task = transcribe_full_task.delay(abs_audio_path)
    
    return jsonify({
        'success': True,
        'message': 'Task submitted',
        'task_id': task.id,
        'status_url': f'/tasks/{task.id}'
    }), 202

@main_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    from .tasks import transcribe_audio_task # Import here to avoid circular imports? No, tasks.py imports celery
    # Actually we can inspect result using AsyncResult
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
        # something went wrong in the background job
        response['error'] = str(task_result.info)
        
    return jsonify(response)
