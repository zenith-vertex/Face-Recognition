# Face Recognition System

A Python/FastAPI face recognition system for attendance and access logging.

## Liveness / Anti-Spoofing Decision

This system is scoped as an **attendance/access-logging** tool for moderate-stakes use cases.
Liveness detection (anti-spoofing) is **not included in the current MVP**. A printed photo or
screen replay can currently bypass the system. If this software is ever used to gate physical
access or high-stakes actions, liveness must be added before deployment.

## Biometric Data Retention & Destruction Policy

- Enrolled embeddings are retained until the user is deleted or the user's faces are deleted.
- Recognition logs are retained indefinitely for audit purposes unless explicitly purged.
- Users (or their administrators) may request deletion at any time via `DELETE /users/{id}/faces`.
- The deletion endpoint actually removes embeddings and logs the action in `biometric_access_audit`.
- There is no automated retention purge in this version; add one if required by your institution.

## Setup

1. Clone the repo and create a virtual environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. Copy `.env.example` to `.env` and update values. Ensure `EMBEDDING_ENCRYPTION_KEY` is set
    to a base64-encoded 32-byte key (use `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).

3. Start PostgreSQL with pgvector:
    ```bash
    docker compose -f docker/docker-compose.yml up -d db
    ```

4. Run migrations:
    ```bash
    alembic upgrade head
    ```

5. Start the app:
    ```bash
    uvicorn app.main:app --reload
    ```

## Threshold Calibration

Do **not** use the default threshold in production. Run the calibration script against your own
validation set to pick an operating point:

```bash
python scripts/calibrate_threshold.py
```

This computes FAR/FRR curves from genuine/impostor pairs in your enrolled population and
suggests a threshold.

## API

- `POST /auth/login` - login (form params `username`, `password`)
- `POST /users` - create user (admin/instructor)
- `POST /users/{id}/consent` - grant biometric consent
- `POST /users/{id}/faces` - enroll face image
- `DELETE /users/{id}/faces` - delete all biometric data
- `POST /recognize` - recognize face from image upload
- `GET /logs` - list recognition logs
- `GET /audit` - biometric access audit trail (admin)

## License

MIT

