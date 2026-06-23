import argparse
import os
import numpy as np
from PIL import Image

from detectors import HaarDetector, MTCNNDetector, YOLOv8Detector
from embedders import DlibEmbedder, FaceNetEmbedder, ArcFaceEmbedder


def get_detector(name: str):
    detectors = {
        "haar": HaarDetector,
        "mtcnn": MTCNNDetector,
        "yolov8": YOLOv8Detector
    }
    return detectors[name]()


def get_embedder(name: str):
    embedders = {
        "dlib": DlibEmbedder,
        "facenet": FaceNetEmbedder,
        "arcface": ArcFaceEmbedder
    }
    return embedders[name]()


def crop_face(frame, bbox, margin_ratio=0.15):
    x, y, w, h = bbox
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    return frame[y1:y2, x1:x2]


def enroll_person(name: str, source_dir: str, detector, embedder):
    all_embeddings = []
    
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        filepath = os.path.join(source_dir, filename)
        frame = np.array(Image.open(filepath))
        
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        
        detections = detector.detect(frame)
        
        for det in detections:
            bbox = det["bbox"]
            x, y, w, h = bbox
            
            if w == 0 or h == 0:
                continue
            
            face_crop = crop_face(frame, bbox, 0.15)
            embedding = embedder.embed(face_crop)
            
            if not np.allclose(embedding, 0):
                all_embeddings.append(embedding)
    
    if not all_embeddings:
        print(f"Warning: No valid face embeddings extracted for {name}")
        return None
    
    avg_embedding = np.mean(all_embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
    
    return avg_embedding


def main():
    parser = argparse.ArgumentParser(description="Enroll a person for face recognition")
    parser.add_argument("name", help="Name of the person to enroll")
    parser.add_argument("--source", required=True, help="Folder containing photos of the person")
    parser.add_argument("--detector", default="haar", help="haar | mtcnn | yolov8")
    parser.add_argument("--embedder", default="dlib", help="dlib | facenet | arcface")
    args = parser.parse_args()
    
    detector = get_detector(args.detector)
    embedder = get_embedder(args.embedder)
    
    if not os.path.isdir(args.source):
        print(f"Error: Source directory not found: {args.source}")
        return
    
    cache_dir = os.path.join(os.path.dirname(__file__), "embeddings_cache", args.embedder)
    os.makedirs(cache_dir, exist_ok=True)
    
    embedding = enroll_person(args.name, args.source, detector, embedder)
    
    if embedding is not None:
        cache_path = os.path.join(cache_dir, f"{args.name}.npy")
        np.save(cache_path, embedding)
        print(f"Enrolled {args.name} with {len(os.listdir(args.source))} photos -> saved to {cache_path}")


if __name__ == "__main__":
    main()