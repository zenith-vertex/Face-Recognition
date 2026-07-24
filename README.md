# AI Face Recognition System

Production-quality, modular Command Line Interface application for face recognition built with Python, Clean Architecture, and industry best practices.

## Features

- **Register Users**: Capture and register new users with facial images
- **Webcam Capture**: Real-time face image capture from webcam
- **Face Detection**: Detect faces in images in real time
- **Embedding Generation**: Generate 128-dimensional facial embeddings
- **Model Training**: Build and train the recognition database
- **Face Recognition**: Recognize registered users from camera or images
- **Unknown Face Identification**: Flag unrecognized faces
- **User Management**: List, view, and delete registered users
- **Log Export**: Export recognition logs to CSV or JSON
- **CLI Interface**: Complete terminal-based workflow

## Architecture

The application follows **Clean Architecture** with clear separation of concerns:

```
src/
├── core/                 # Domain layer (entities, ports, domain services)
├── application/          # Application layer (use cases/interactors)
├── infrastructure/       # External adapters (DB, camera, AI engine)
└── presentation/         # CLI interface
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CMake and Visual Studio Build Tools (Windows) or build-essential (Linux)
- A webcam (for live capture features)

### Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and adjust settings:

```bash
cp .env.example .env
```

Key settings:
- `DATABASE_PATH`: SQLite database location
- `ENCODINGS_DIR`: Directory to store face embeddings
- `RECOGNITION_TOLERANCE`: Threshold for face matching (lower = stricter)
- `CAMERA_SOURCE`: Webcam index (0 for default)

## Usage

```bash
face-recognition register "John Doe" --images "path/to/images"
face-recognition capture "John Doe" --count 10
face-recognition train
face-recognition recognize
face-recognition unknown
face-recognition list-users
face-recognition delete-user "John Doe"
face-recognition logs --limit 20
face-recognition export-logs --output logs.csv --format csv
face-recognition clean-unknowns
```

## Commands

| Command | Description |
|---------|-------------|
| `register` | Register a new user from images |
| `capture` | Capture images from webcam for a user |
| `train` | Train/update the recognition model |
| `recognize` | Start live recognition from webcam |
| `unknown` | Identify and isolate unknown faces |
| `list-users` | List all registered users |
| `delete-user` | Remove a user and their data |
| `logs` | View recent recognition logs |
| `export-logs` | Export logs to CSV or JSON |
| `clean-unknowns` | Clear unknown face entries |

## Technology Stack

- **Language**: Python 3.10+
- **CLI Framework**: Click
- **Face Recognition**: face_recognition (dlib)
- **Camera**: OpenCV (cv2)
- **Database**: SQLite3
- **Output**: Rich
- **Validation**: Pydantic

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - see LICENSE file for details.
