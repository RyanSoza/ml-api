"""HTTP API for the model"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from ml_api.config import Settings
from ml_api.predict import load_model, predict
from ml_api.schemas import PredictRequest, PredictResponse

logger = logging.getLogger("ml_api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs before first request
        app.state.model = load_model(settings.artifact_path)
        yield
        # Runs at shutdown

    app = FastAPI(title="ml_api", version=settings.version, lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s %s %.1fms", request.method, request.url.path, response.status_code, elapsed_ms
        )

        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": settings.version, "service": "ml-api"}

    @app.post("/predict", response_model=PredictResponse)
    def predict_one(body: PredictRequest, request: Request) -> PredictResponse:
        model = request.app.state.model
        prediction = predict(model, body.to_frame())

        return PredictResponse(prediction=prediction[0], version=settings.version)

    return app


app = create_app()
