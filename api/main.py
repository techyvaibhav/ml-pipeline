"""FastAPI app: model is loaded once at startup."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPrediction,
    BatchRequest,
    Customer,
    Health,
    ModelInfo,
    SinglePrediction,
)
from src.config import MODEL_PATH
from src.predict import load_bundle, predict_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("churn-api")  # app logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup (app keeps running even if it is missing)."""
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))  # model file path
    app.state.path = path
    try:
        app.state.bundle = load_bundle(path)
        log.info("model loaded: version=%s", app.state.bundle["version"])
    except Exception:
        app.state.bundle = None
        log.exception("could not load model from %s", path)
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def get_bundle(request: Request) -> dict:
    """Return loaded model or raise 503."""
    b = request.app.state.bundle  # model bundle
    if b is None:
        raise HTTPException(503, "Model not loaded. Train one with `make train` and restart.")
    return b


@app.exception_handler(RequestValidationError)
async def on_422(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Clear, compact 422 messages."""
    errs = [{"field": ".".join(str(x) for x in e["loc"][1:]) or "body", "message": e["msg"]}
            for e in exc.errors()]  # simplified errors
    return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errs})


@app.exception_handler(Exception)
async def on_500(_: Request, exc: Exception) -> JSONResponse:
    """Log and return a clean 500."""
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=Health)
def health(request: Request):
    """Liveness + model status (503 if model missing, so docker marks it unhealthy)."""
    ok = request.app.state.bundle is not None  # model loaded?
    body = {"status": "ok" if ok else "model_missing", "model_loaded": ok}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request):
    """Version, test metrics and input features."""
    b = get_bundle(request)  # model bundle
    return ModelInfo(version=b["version"], trained_at=b["trained_at"], algorithm=b["algorithm"],
                     threshold=b["threshold"], metrics=b["metrics"], features=b["features"])


@app.post("/predict", response_model=SinglePrediction)
def predict(cust: Customer, request: Request):
    """Churn probability + label for one customer."""
    b = get_bundle(request)  # model bundle
    res = predict_batch([cust.model_dump()], request.app.state.path)[0]  # prediction
    log.info("predict prob=%.3f label=%s", res["churn_probability"], res["label"])
    return {**res, "version": b["version"]}


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_many(req: BatchRequest, request: Request):
    """Churn predictions for many customers (same order)."""
    b = get_bundle(request)  # model bundle
    res = predict_batch([c.model_dump() for c in req.customers], request.app.state.path)
    log.info("batch predict n=%d", len(res))
    return {"version": b["version"], "count": len(res), "predictions": res}
