import os
import numpy as np
from repository import (
    create_person, add_face_embedding, find_nearest_match,
    log_recognition, get_recognition_logs, get_vector_backend, VECTOR_DIMENSIONS
)
from database import Base, engine


def generate_distinct_test_vector(person_id: int, dim: int = 128):
    vec = np.zeros(dim, dtype=np.float32)
    vec[person_id % dim] = 1.0
    return vec


def main():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created.")

    tests_passed = 0
    tests_failed = 0

    # Step 1: Create 3 dummy persons
    print("\n=== Step 1: Create 3 persons ===")
    try:
        person_ids = []
        for i, (name, role) in enumerate([("Alice", "admin"), ("Bob", "user"), ("Charlie", "guest")]):
            pid = create_person(name=name, role=role)
            person_ids.append(pid)
            print(f"Created person {name} with id {pid}")
        print("PASS: All 3 persons created")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        tests_failed += 1

    # Step 2: Insert embeddings
    print("\n=== Step 2: Insert embeddings for each person ===")
    try:
        embedder = "dlib"
        dim = VECTOR_DIMENSIONS[embedder]
        emb_ids = []
        for i, pid in enumerate(person_ids):
            vec = generate_distinct_test_vector(i, dim=dim)
            eid = add_face_embedding(person_id=pid, embedding=vec, embedder_model=embedder, source_image_path=f"test{i}.jpg")
            emb_ids.append(eid)
            print(f"Added embedding with id {eid} for person {i}")
        print("PASS: Embeddings inserted")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        tests_failed += 1

    # Step 3: Find nearest match
    print("\n=== Step 3: Find nearest match ===")
    try:
        query_vec = generate_distinct_test_vector(1, dim=128).copy()
        query_vec[1] += 0.1
        query_vec = query_vec / np.linalg.norm(query_vec)

        results = find_nearest_match(query_vec, embedder_model="dlib", metric="cosine", top_k=1)
        print(f"Nearest match results: {results}")

        if results and results[0]["person_id"] == person_ids[1]:
            print(f"PASS: Nearest match returned correct person (Bob, id={person_ids[1]})")
            tests_passed += 1
        else:
            actual_id = results[0]["person_id"] if results else "None"
            print(f"FAIL: Expected person_id {person_ids[1]}, got {actual_id}")
            tests_failed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        tests_failed += 1

    # Step 4: Log match event
    print("\n=== Step 4: Log match recognition ===")
    try:
        log_id = log_recognition(
            matched_person_id=person_ids[0],
            confidence_score=0.85,
            metric_used="cosine",
            decision="match",
            detector_used="haar",
            embedder_used="dlib",
            source_camera="webcam"
        )
        print(f"PASS: Match log created with id {log_id}")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        tests_failed += 1

    # Step 5: Log no_match event
    print("\n=== Step 5: Log no_match recognition ===")
    try:
        log_id = log_recognition(
            matched_person_id=None,
            confidence_score=0.42,
            metric_used="cosine",
            decision="no_match",
            detector_used="haar",
            embedder_used="dlib",
            source_camera="webcam"
        )
        print(f"PASS: No-match log created with id {log_id}")
        tests_passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        tests_failed += 1

    # Step 6: Retrieve logs
    print("\n=== Step 6: Retrieve recognition logs ===")
    try:
        all_logs = get_recognition_logs()
        match_logs = get_recognition_logs(decision="match")
        no_match_logs = get_recognition_logs(decision="no_match")

        if len(all_logs) >= 2 and len(match_logs) >= 1 and len(no_match_logs) >= 1:
            print(f"PASS: Retrieved logs - total: {len(all_logs)}, match: {len(match_logs)}, no_match: {len(no_match_logs)}")
            tests_passed += 1
        else:
            print(f"FAIL: Expected at least 2 logs, got total: {len(all_logs)}, match: {len(match_logs)}, no_match: {len(no_match_logs)}")
            tests_failed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        tests_failed += 1

    print(f"\n{'='*40}")
    print(f"Tests passed: {tests_passed}/6")
    print(f"Tests failed: {tests_failed}/6")

    if tests_failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    main()