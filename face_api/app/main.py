from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from .config import settings
from .services.pipeline_service import PipelineService, VECTOR_DIMENSIONS


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = PipelineService(
        detector_name=settings.ACTIVE_DETECTOR,
        embedder_name=settings.ACTIVE_EMBEDDER,
        threshold=settings.MATCH_THRESHOLD,
        max_image_dimension=settings.MAX_IMAGE_DIMENSION
    )

    try:
        pipeline.initialize()
        print(f"Pipeline initialized: detector={settings.ACTIVE_DETECTOR}, embedder={settings.ACTIVE_EMBEDDER}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize pipeline: {e}")

    app.state.pipeline = pipeline

    yield

    print("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Recognition API",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    from .routers import persons, recognize, logs, streams
    app.include_router(persons.router)
    app.include_router(recognize.router)
    app.include_router(logs.router)
    app.include_router(streams.router)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


app = create_app()