import argparse
import json
import time
import os
import cv2
import numpy as np
import yaml
from datetime import datetime

from detectors import HaarDetector, MTCNNDetector, YOLOv8Detector
from embedders import DlibEmbedder, FaceNetEmbedder, ArcFaceEmbedder
from matcher import match


def get_detector(name: str):
    detectors = {
        "haar": HaarDetector,
        "mtcnn": MTCNNDetector,
        "yolov8": YOLOv8Detector
    }
    if name not in detectors:
        raise ValueError(f"Unknown detector: {name}. Available: {list(detectors.keys())}")
    return detectors[name]()


def get_embedder(name: str):
    embedders = {
        "dlib": DlibEmbedder,
        "facenet": FaceNetEmbedder,
        "arcface": ArcFaceEmbedder
    }
    if name not in embedders:
        raise ValueError(f"Unknown embedder: {name}. Available: {list(embedders.keys())}")
    return embedders[name]()


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_embeddings_cache(embedder_name: str):
    cache_dir = os.path.join(os.path.dirname(__file__), "embeddings_cache", embedder_name)
    embeddings = {}
    
    if not os.path.exists(cache_dir):
        return embeddings
    
    for filename in os.listdir(cache_dir):
        if filename.endswith(".npy"):
            name = filename[:-4]
            filepath = os.path.join(cache_dir, filename)
            embeddings[name] = np.load(filepath)
    
    return embeddings


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


def process_frame(frame, detector, embedder, known_embeddings, threshold, metric, margin_ratio):
    results = detector.detect(frame)
    output = frame.copy()
    
    for det in results:
        bbox = det["bbox"]
        x, y, w, h = bbox
        
        if w == 0 or h == 0:
            continue
        
        detect_latency = det["latency_ms"]
        face_crop = crop_face(frame, bbox, margin_ratio)
        
        embed_start = time.perf_counter()
        query_emb = embedder.embed(face_crop)
        embed_latency = (time.perf_counter() - embed_start) * 1000
        
        match_result = match(query_emb, known_embeddings, metric, threshold)
        total_latency = detect_latency + embed_latency
        
        name = match_result["name"]
        score = match_result["score"]
        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": "webcam",
            "detector": detector.__class__.__name__.replace("Detector", "").lower(),
            "embedder": embedder.__class__.__name__.replace("Embedder", "").lower(),
            "bbox": [x, y, w, h],
            "matched_name": name,
            "score": score,
            "metric": metric,
            "decision": "match" if name != "Unknown" else "no_match",
            "detect_latency_ms": detect_latency,
            "embed_latency_ms": embed_latency,
            "total_latency_ms": total_latency
        }
        
        yield output, log_entry


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Pipeline")
    parser.add_argument("--source", default="webcam", help="webcam, camera index, or path to image/video")
    parser.add_argument("--detector", default="haar", help="haar | mtcnn | yolov8")
    parser.add_argument("--embedder", default="dlib", help="dlib | facenet | arcface")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold from config")
    parser.add_argument("--metric", default="cosine", help="cosine | euclidean")
    args = parser.parse_args()
    
    config = load_config()
    
    metric = args.metric
    threshold = args.threshold
    if threshold is None:
        threshold = config["thresholds"].get(metric, 0.6)
    
    detector = get_detector(args.detector)
    embedder = get_embedder(args.embedder)
    known_embeddings = load_embeddings_cache(args.embedder)
    
    if known_embeddings:
        print(f"Loaded {len(known_embeddings)} enrolled faces for {args.embedder}")
    else:
        print("No enrolled faces found. Run enroll.py first.")
    
    log_file = None
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if args.source == "webcam" or (args.source.isdigit() if isinstance(args.source, str) else False):
        camera_idx = int(args.source) if args.source.isdigit() else 0
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_idx}")
            return
        
        log_file = open(os.path.join(results_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"), "w")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from camera")
                break
            
            for output, log_entry in process_frame(frame, detector, embedder, known_embeddings, threshold, metric, config.get("margin_ratio", 0.15)):
                cv2.imshow("Face Recognition", output)
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
    else:
        if not os.path.exists(args.source):
            print(f"Error: Source file not found: {args.source}")
            return
        
        if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                print(f"Error: Cannot open video file {args.source}")
                return
            
            log_file = open(os.path.join(results_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"), "w")
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                for output, log_entry in process_frame(frame, detector, embedder, known_embeddings, threshold, metric, config.get("margin_ratio", 0.15)):
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                
                frame_idx += 1
            
            cap.release()
            if log_file:
                log_file.close()
            
            print(f"Processed {frame_idx} frames from video")
        else:
            frame = cv2.imread(args.source)
            if frame is None:
                print(f"Error: Cannot read image file {args.source}")
                return
            
            log_file = open(os.path.join(results_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"), "w")
            
            last_output = frame.copy()
            for output, log_entry in process_frame(frame, detector, embedder, known_embeddings, threshold, metric, config.get("margin_ratio", 0.15)):
                last_output = output
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
            
            log_file.close()
            cv2.imwrite(os.path.join(results_dir, "annotated_output.jpg"), last_output)
            print(f"Processed image, saved to results/annotated_output.jpg")


if __name__ == "__main__":
    main()