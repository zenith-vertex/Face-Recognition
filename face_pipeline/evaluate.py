import argparse
import os
import time
import numpy as np
import csv

import cv2
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


def evaluate_combination(detector_name, embedder_name, threshold, metric, test_data_dir):
    detector = get_detector(detector_name)
    embedder = get_embedder(embedder_name)
    known_embeddings = load_embeddings_cache(embedder_name)
    
    if not known_embeddings:
        return None
    
    stats = {
        "total_faces": 0,
        "true_accepts": 0,
        "false_accepts": 0,
        "false_rejects": 0,
        "total_detect_latency_ms": 0.0,
        "total_embed_latency_ms": 0.0,
        "total_latency_ms": 0.0
    }
    
    margin_ratio = 0.15
    
    for person_name in os.listdir(test_data_dir):
        person_dir = os.path.join(test_data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            filepath = os.path.join(person_dir, filename)
            frame = cv2.imread(filepath)
            if frame is None:
                continue
            
            stats["total_faces"] += 1
            detections = detector.detect(frame)
            
            for det in detections:
                bbox = det["bbox"]
                x, y, w, h = bbox
                
                if w == 0 or h == 0:
                    continue
                
                stats["total_detect_latency_ms"] += det["latency_ms"]
                
                face_crop = crop_face(frame, bbox, margin_ratio)
                
                embed_start = time.perf_counter()
                query_emb = embedder.embed(face_crop)
                embed_latency = (time.perf_counter() - embed_start) * 1000
                stats["total_embed_latency_ms"] += embed_latency
                
                match_result = match(query_emb, known_embeddings, metric, threshold)
                stats["total_latency_ms"] += det["latency_ms"] + embed_latency
                
                if match_result["name"] == person_name:
                    stats["true_accepts"] += 1
                elif match_result["name"] != "Unknown":
                    stats["false_accepts"] += 1
                else:
                    stats["false_rejects"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition pipeline combinations")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold from config")
    parser.add_argument("--metric", default="cosine", help="cosine | euclidean")
    args = parser.parse_args()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    
    detectors = ["haar", "mtcnn", "yolov8"]
    embedders = ["dlib", "facenet", "arcface"]
    
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    threshold = args.threshold
    metric = args.metric
    
    if threshold is None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        threshold = config["thresholds"].get(metric, 0.6)
    
    results = []
    for det in detectors:
        for emb in embedders:
            stats = evaluate_combination(det, emb, threshold, metric, test_data_dir)
            if stats:
                avg_detect = stats["total_detect_latency_ms"] / max(stats["total_faces"], 1)
                avg_total = stats["total_latency_ms"] / max(stats["total_faces"], 1)
                
                results.append({
                    "detector": det,
                    "embedder": emb,
                    "total_faces": stats["total_faces"],
                    "true_accept_rate": stats["true_accepts"] / max(stats["total_faces"], 1),
                    "false_accept_rate": stats["false_accepts"] / max(stats["total_faces"], 1),
                    "false_reject_rate": stats["false_rejects"] / max(stats["total_faces"], 1),
                    "avg_detect_latency_ms": avg_detect,
                    "avg_total_latency_ms": avg_total
                })
    
    csv_path = os.path.join(results_dir, "evaluation_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["detector", "embedder", "total_faces",
                                                "true_accept_rate", "false_accept_rate",
                                                "false_reject_rate", "avg_detect_latency_ms",
                                                "avg_total_latency_ms"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Evaluation complete. Report saved to {csv_path}")
    for r in results:
        print(f"{r['detector']}/{r['embedder']}: TAR={r['true_accept_rate']:.2%}, FAR={r['false_accept_rate']:.2%}, FRR={r['false_reject_rate']:.2%}")


if __name__ == "__main__":
    main()