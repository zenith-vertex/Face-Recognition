# Face Recognition Database Layer (Phase 1)

Persistent storage for the face recognition pipeline using PostgreSQL with pgvector support.

## Setup

### 1. Start PostgreSQL with pgvector (Docker)

```bash
cd face_db
docker-compose up -d
```

This starts a PostgreSQL 16 container with pgvector extension pre-installed.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update DATABASE_URL if needed:

```bash
copy .env.example .env
```

### 4. Run Migrations

```bash
alembic upgrade head
```

Or for direct table creation (for testing):
```bash
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```

## Usage

```bash
# Seed sample data
python seed_data.py

# Run validation tests
python test_db_pipeline.py
```

## Repository Interface

```python
from repository import (
    create_person, get_person, deactivate_person,
    add_face_embedding, get_embeddings_for_person,
    find_nearest_match, log_recognition, get_recognition_logs
)

# Create a person
person_id = create_person("Alice", "admin")

# Add embedding (dlib=128 dim, facenet/arcface=512 dim)
add_face_embedding(person_id, embedding_vector, "dlib", "photo.jpg")

# Find match
results = find_nearest_match(query_vector, "dlib", "cosine")

# Log recognition
log_recognition(person_id, 0.85, "cosine", "match", "haar", "dlib")
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:password@localhost:5432/face_recognition_db` |
| `VECTOR_BACKEND` | `pgvector` or `array` (fallback) | `pgvector` |

### Embedder Model & Vector Dimension

This layer uses **dlib** (128 dimensions) as the production-bound embedder. The vector column is `VECTOR(128)`.

Embeddings are validated at insertion time to match the 128-dimension requirement.

## Database Schema

- **persons** - User records with registration info and status
- **face_embeddings** - Face vectors linked to persons, with HNSW index for ANN search
- **recognition_logs** - Recognition events for audit/recall

## Testing

The `test_db_pipeline.py` script validates:
1. Person creation with distinct roles
2. Face embedding insertion with dimension validation
3. Nearest-neighbor similarity search returns correct matches
4. Recognition logging (match and no_match decisions)
5. Log retrieval with filtering
6. Constraint enforcement (decision ↔ matched_person_id linkage)