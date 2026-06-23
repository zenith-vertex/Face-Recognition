import numpy as np
from repository import create_person, add_face_embedding, VECTOR_DIMENSIONS


def generate_random_embedding(dim: int = 128):
    vec = np.random.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def main():
    print("Seeding sample persons and embeddings...")

    persons = [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "user"},
        {"name": "Charlie", "role": "guest"}
    ]

    for p in persons:
        pid = create_person(name=p["name"], role=p["role"])
        print(f"Created person {p['name']} with id {pid}")

        emb = generate_random_embedding(dim=128)
        add_face_embedding(person_id=pid, embedding=emb, embedder_model="dlib", source_image_path=f"seed_{p['name'].lower()}.jpg")
        print(f"  Added dlib embedding (dim=128)")

    print("Seed data complete.")


if __name__ == "__main__":
    main()