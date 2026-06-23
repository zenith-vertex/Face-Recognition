FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY face_pipeline/ /app/face_pipeline/
COPY face_db/ /app/face_db/
COPY face_api/ /app/face_api/
COPY requirements.txt /app/requirements.txt
COPY face_pipeline/config.yaml /app/face_pipeline/config.yaml

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV DATABASE_URL=postgresql://postgres:password@db:5432/face_recognition_db

WORKDIR /app/face_api

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]