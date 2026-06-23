# Face Recognition Core Pipeline (Phase 0)

Offline, testable Python pipeline for face detection + embedding + matching.

## Setup

```bash
pip install -r requirements.txt
```

### OS-level Dependencies for dlib

**Windows:**
- Install Visual Studio Build Tools with C++ support
- Install cmake: `pip install cmake`
- Then: `pip install dlib`

**macOS:**
- `brew install cmake`
- `pip install dlib`

**Linux (Ubuntu/Debian):**
- `sudo apt-get install build-essential cmake`
- `pip install dlib`

## Usage

### Enroll a person
```bash
python enroll.py "John Doe" --source ./known_faces/john_doe --detector haar --embedder dlib
```

### Run real-time webcam recognition
```bash
python main.py --source webcam --detector haar --embedder dlib
python main.py --source webcam --detector yolov8 --embedder arcface --metric euclidean --threshold 0.4
```

### Process an image
```bash
python main.py --source path/to/image.jpg --detector mtcnn --embedder facenet
```

### Process a video
```bash
python main.py --source path/to/video.mp4 --detector yolov8 --embedder dlib
```

### Evaluate all combinations
```bash
python evaluate.py --metric cosine
```

## CLI Arguments

- `--source`: `webcam` (or camera index 0,1,2...), or path to image/video file
- `--detector`: `haar | mtcnn | yolov8`
- `--embedder`: `dlib | facenet | arcface`
- `--threshold`: optional float override (cosine default: 0.6, euclidean default: 0.4)
- `--metric`: `cosine | euclidean`

## Project Structure

```
face_pipeline/
├── main.py              # CLI entry point
├── config.yaml          # thresholds, model paths
├── detectors/           # detection backends
│   ├── base.py
│   ├── haar_detector.py
│   ├── mtcnn_detector.py
│   └── yolov8_detector.py
├── embedders/           # embedding backends
│   ├── base.py
│   ├── dlib_embedder.py
│   ├── facenet_embedder.py
│   └── arcface_embedder.py
├── matcher.py           # distance/similarity logic
├── enroll.py            # register known persons
├── known_faces/         # enrolled person photos
├── embeddings_cache/    # cached .npy embeddings
├── test_data/           # labeled test images
└── results/             # logs and reports
```

## Swapping Backends

Edit the `--detector` and `--embedder` arguments to test different combinations. Each component implements a base abstract class for easy replacement:

- Detectors: Extend `BaseDetector` with `detect(frame) -> list[dict]`
- Embedders: Extend `BaseEmbedder` with `embed(face_crop) -> np.ndarray`