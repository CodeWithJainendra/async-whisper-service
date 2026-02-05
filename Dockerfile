FROM python:3.9-slim

# Install system dependencies for OpenAI Whisper (ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 5011

# Command is overridden in docker-compose usually, but default to API
CMD ["python", "run.py"]
