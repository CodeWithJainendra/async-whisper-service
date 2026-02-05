import os

class Config:
    broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', os.path.join(os.getcwd(), 'chroma_db'))
