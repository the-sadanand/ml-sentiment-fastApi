"""FastAPI application serving a sentiment analysis model."""
import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Logging (stdout, INFO by default)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("sentiment_api")

# Global model placeholder - loaded exactly once at startup
ml_model: Optional[Any] = None


def load_model() -> Any:
    """Load the serialized model from MODEL_PATH."""
    model_path = os.getenv("MODEL_PATH", "./model/sentiment_model.pkl")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)  # noqa: S301 - trusted, self-generated artifact
    logger.info("Model loaded successfully from %s", model_path)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    try:
        ml_model = load_model()
    except Exception:
        ml_model = None
        logger.error("Failed to load model", exc_info=True)
    yield
    ml_model = None
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready REST API serving a TF-IDF + Logistic Regression sentiment model.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyse")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class PredictionOutput(BaseModel):
    sentiment: str
    score: float


# --------------------------------------------------------------------------- #
# Error handling - never leak stack traces to clients
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
    # Return a safe summary (location + message only, no raw input echoed back)
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(input_data: TextInput):
    snippet = input_data.text[:50]
    logger.info("Request received: POST /predict text=%r", snippet)

    if ml_model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    try:
        probabilities = ml_model.predict_proba([input_data.text])[0]
        best_idx = int(probabilities.argmax())
        sentiment = str(ml_model.classes_[best_idx])
        score = round(float(probabilities[best_idx]), 4)
    except Exception:
        logger.error("Model inference failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    logger.info("Prediction successful: text=%r sentiment=%s score=%.4f", snippet, sentiment, score)
    return {"sentiment": sentiment, "score": score}
