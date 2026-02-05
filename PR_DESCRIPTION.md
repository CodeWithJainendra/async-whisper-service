# Refactor: Async Transcription Architecture

## Summary
Refactoring the existing monolithic Flask application into a scalable, asynchronous architecture using Celery and Redis. This change addresses the performance bottleneck where long-running transcription tasks blocked the HTTP request thread.

## Key Changes
- **Async Processing**: Implemented Celery for background task processing.
- **Message Broker**: Integrated Redis as the message broker and result backend.
- **Dockerization**: Added `Dockerfile` and `docker-compose.yml` for fully reproducible local development and deployment.
- **Structural Refactor**: Moved from single `app.py` to a modular application factory pattern (`app/__init__.py`, `app/routes.py`, `app/tasks.py`).
- **Resilience**: Added error handling and state updates (PENDING, PROCESSING, SUCCESS) for better UX.

## Technical Details
- **Flask**: Updated to use Blueprints and Application Factory pattern.
- **Whisper**: Model loading is now handled within the Celery worker process to avoid blocking the API server.
- **API**:
    - `POST /asr`: Returns `202 Accepted` immediately with a `task_id`.
    - `GET /tasks/<task_id>`: Endpoint to poll for status and results.
    - `POST /asr/full`: Supports translation and transcription.

## How to Test
1. Ensure Docker is installed.
2. Run `docker-compose up --build`.
3. Send a POST request to `http://localhost:5011/asr` with an audio file.
4. Use the returned `status_url` to check progress.
