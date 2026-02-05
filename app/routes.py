"""
API Routes for Audio Transcription Service

FIXES IMPLEMENTED:
1. Fixed search_audio_task sync call - now uses search_audio_sync
2. Added proper error handling with logging
3. Added /health endpoint for monitoring
4. Added request validation
5. Improved response formatting
"""

from flask import Blueprint, request, jsonify, render_template, current_app
import tempfile
import os
import logging
from .tasks import (
    transcribe_and_embed_task, 
    search_audio_sync,  # FIX: Use sync function instead of task
    analyze_transcript_task,
    ModelManager
)

# Configure logging
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@main_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.
    
    FIX: Added to detect system issues early.
    Returns status of all critical components.
    """
    try:
        health = ModelManager.health_check()
        
        # Check Redis connectivity
        try:
            from . import celery
            celery.control.ping(timeout=2)
            health["redis"] = {"connected": True}
        except Exception as e:
            health["redis"] = {"connected": False, "error": str(e)}
        
        # Determine overall status
        all_healthy = (
            health.get("whisper", {}).get("healthy", False) or health.get("whisper", {}).get("loaded") is None
        ) and (
            health.get("embedding", {}).get("healthy", False) or health.get("embedding", {}).get("loaded") is None
        ) and (
            health.get("chromadb", {}).get("connected", False)
        )
        
        status_code = 200 if all_healthy else 503
        health["status"] = "healthy" if all_healthy else "degraded"
        
        return jsonify(health), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@main_bp.route('/asr', methods=['POST'])
def transcribe():
    """
    Submit audio for transcription.
    
    Accepts multipart form with:
    - audio: The audio file (required)
    - translate: Whether to translate to English (optional, default: false)
    
    Returns task ID for polling status.
    """
    if 'audio' not in request.files:
        logger.warning("Transcription request missing audio file")
        return jsonify({
            'success': False, 
            'message': 'No audio file uploaded',
            'error_code': 'MISSING_AUDIO'
        }), 400

    audio_file = request.files['audio']
    
    # Validate file
    if audio_file.filename == '':
        return jsonify({
            'success': False,
            'message': 'Empty filename',
            'error_code': 'EMPTY_FILENAME'
        }), 400
    
    translate_to_english = request.form.get('translate', 'false').lower() == 'true'
    
    # Ensure upload directory exists
    upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    import uuid
    original_ext = os.path.splitext(audio_file.filename)[1] or '.webm'
    filename = f"{uuid.uuid4()}{original_ext}"
    audio_path = os.path.join(upload_dir, filename)
    
    try:
        audio_file.save(audio_path)
        logger.info(f"Saved audio file: {filename}")
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to save audio file',
            'error_code': 'SAVE_FAILED'
        }), 500

    abs_audio_path = os.path.abspath(audio_path)

    # Submit to Celery
    try:
        task = transcribe_and_embed_task.delay(abs_audio_path, translate_to_english)
        logger.info(f"Submitted transcription task: {task.id}")
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        # Clean up the file since task wasn't submitted
        try:
            os.remove(abs_audio_path)
        except:
            pass
        return jsonify({
            'success': False,
            'message': 'Failed to submit processing task',
            'error_code': 'TASK_SUBMIT_FAILED'
        }), 503
    
    return jsonify({
        'success': True,
        'message': 'Task submitted (Transcription + Indexing)',
        'task_id': task.id,
        'status_url': f'/tasks/{task.id}',
        'translate': translate_to_english
    }), 202


@main_bp.route('/search', methods=['GET'])
def search():
    """
    Search indexed audio transcripts.
    
    FIX: Uses search_audio_sync instead of search_audio_task
    to avoid Celery decorator issues when called synchronously.
    
    Query params:
    - q: Search query (required)
    - n: Number of results (optional, default: 5, max: 20)
    """
    query = request.args.get('q')
    if not query:
        return jsonify({
            'success': False,
            'error': 'Missing query parameter "q"',
            'error_code': 'MISSING_QUERY'
        }), 400
    
    if len(query) < 2:
        return jsonify({
            'success': False,
            'error': 'Query too short (minimum 2 characters)',
            'error_code': 'QUERY_TOO_SHORT'
        }), 400
    
    # Parse and validate n_results
    try:
        n_results = int(request.args.get('n', 5))
        n_results = max(1, min(20, n_results))  # Clamp between 1 and 20
    except ValueError:
        n_results = 5
    
    logger.info(f"Search request: query='{query}', n={n_results}")
    
    try:
        # FIX: Call sync function directly, not the Celery task
        results = search_audio_sync(query, n_results)
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        })
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'error_code': 'SEARCH_FAILED'
        }), 500


@main_bp.route('/analyze/<task_id>', methods=['POST'])
def analyze_task(task_id):
    """
    Start AI analysis on a completed transcription.
    
    The transcription task must be complete before analysis can start.
    """
    from . import celery
    
    if not task_id:
        return jsonify({
            'success': False, 
            'message': 'Missing task_id',
            'error_code': 'MISSING_TASK_ID'
        }), 400
    
    try:
        task_result = celery.AsyncResult(task_id)
    except Exception as e:
        logger.error(f"Failed to get task result: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to retrieve task',
            'error_code': 'TASK_RETRIEVAL_FAILED'
        }), 500
    
    # Check task status
    if task_result.state == 'PENDING':
        return jsonify({
            'success': False, 
            'message': 'Task not found or still pending',
            'error_code': 'TASK_PENDING'
        }), 404
    
    if not task_result.ready():
        return jsonify({
            'success': False, 
            'message': 'Transcription task not finished yet',
            'state': task_result.state,
            'error_code': 'TASK_NOT_READY'
        }), 400
        
    if task_result.failed():
        return jsonify({
            'success': False, 
            'message': 'Transcription task failed',
            'error_code': 'TASK_FAILED'
        }), 400
    
    # Extract transcript
    result = task_result.result
    transcript = result.get('transcript') if isinstance(result, dict) else ""
    
    if not transcript:
        return jsonify({
            'success': False, 
            'message': 'No transcript found to analyze',
            'error_code': 'NO_TRANSCRIPT'
        }), 400
    
    # Start analysis task
    try:
        analysis_task = analyze_transcript_task.delay(transcript)
        logger.info(f"Started analysis task: {analysis_task.id}")
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to start analysis',
            'error_code': 'ANALYSIS_SUBMIT_FAILED'
        }), 503
    
    return jsonify({
        'success': True,
        'message': 'Analysis started',
        'analysis_task_id': analysis_task.id,
        'status_url': f'/tasks/{analysis_task.id}'
    }), 202


@main_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    Get the status of a background task.
    
    Returns current state, progress info, and result when complete.
    """
    from . import celery
    
    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400
    
    try:
        task_result = celery.AsyncResult(task_id)
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        return jsonify({
            'task_id': task_id,
            'status': 'ERROR',
            'error': str(e)
        }), 500
    
    response = {
        'task_id': task_id,
        'status': task_result.status,
    }
    
    if task_result.state == 'PENDING':
        response['result'] = None
        response['message'] = 'Task is waiting to be processed'
    elif task_result.state == 'PROCESSING':
        response['result'] = None
        response['progress'] = task_result.info if task_result.info else {}
    elif task_result.state == 'SUCCESS':
        response['result'] = task_result.result
    elif task_result.state == 'FAILURE':
        response['error'] = str(task_result.info) if task_result.info else 'Unknown error'
        response['result'] = None
    else:
        # Handle other states (RETRY, REVOKED, etc.)
        response['result'] = task_result.info if task_result.info else None
        
    return jsonify(response)


@main_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get statistics about indexed audio.
    
    Returns count of indexed documents and other stats.
    """
    try:
        health = ModelManager.health_check()
        
        return jsonify({
            'success': True,
            'indexed_chunks': health.get('chromadb', {}).get('count', 0),
            'models': {
                'whisper': health.get('whisper', {}),
                'embedding': health.get('embedding', {})
            }
        })
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
