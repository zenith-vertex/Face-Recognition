# Face Recognition API

Phase 2 of the Face Recognition System - FastAPI backend that wraps the detection/embedding pipeline (Phase 0) and database storage layer (Phase 1).

## Environment Setup

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- OpenCV-compatible camera (for streaming) or test images

### Installation

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (copy `.env.example` to `.env`):
```bash
cp .env.example .env
```

3. Configure the `.env` file with your database connection and model settings:
```
DATABASE_URL=postgresql://user:password@localhost:5432/face_recognition_db
ACTIVE_DETECTOR=yolov8
ACTIVE_EMBEDDER=dlib
API_KEY=changeme
MAX_IMAGE_DIMENSION=1280
MATCH_THRESHOLD=0.6
```

4. Ensure the database schema is set up (from Phase 1):
```bash
cd face_db && alembic upgrade head
```

## Running Locally

Start the API server:
```bash
uvicorn app.main:app --reload --port 8000
```

The interactive API documentation will be available at `http://localhost:8000/docs`.

## Running via Docker

1. Build the image:
```bash
docker build -t face-api -f Dockerfile .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://postgres:password@host.docker.internal:5432/face_recognition_db \
  -e ACTIVE_DETECTOR=yolov8 \
  -e ACTIVE_EMBEDDER=dlib \
  face-api
```

## API Endpoints

### Face Registration
- `POST /persons/register-face` - Register a face with one or more images
- `GET /persons` - List all active persons (paginated)
- `GET /persons/{id}` - Get person details with embedding count
- `DELETE /persons/{id}` - Soft delete a person

### Recognition
- `POST /recognize` - Recognize faces in an uploaded image

### Logs
- `GET /logs` - Get recognition logs (filterable by person_id, date, decision)

### Streams
- `POST /streams/start` - Start a live stream (webcam, camera index, or RTSP URL)
- `POST /streams/stop` - Stop a live stream
- `GET /streams/status` - Get status of all active streams

### WebSocket
- `WebSocket /streams/ws/live/{camera_id}` - Real-time recognition events

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Architecture

- Models loaded once at startup via FastAPI lifespan
- Blocking pipeline operations run in thread pool
- Live streams handled in background threads with graceful shutdown
- WebSocket broadcasts use `asyncio.run_coroutine_threadsafe` for thread-safe communication