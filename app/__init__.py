from flask import Flask
from celery import Celery
from .config import Config

celery = Celery(__name__, broker=Config.CELERY_BROKER_URL)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    celery.conf.update(app.config)

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app
