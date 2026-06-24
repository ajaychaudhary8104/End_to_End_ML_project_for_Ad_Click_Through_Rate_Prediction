from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from src.ad_ctr_prediction import logger
from src.ad_ctr_prediction.config.configuration import ConfigurationManager
from src.ad_ctr_prediction.components.inference import ModelInference
from src.ad_ctr_prediction.utils.s3_loader import download_artifacts

# =========================================================
# PATH CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = ROOT_DIR / "templates"

INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"


# =========================================================
# LOAD HTML TEMPLATE
# =========================================================

def load_index_html() -> str:

    if not INDEX_HTML_PATH.exists():

        raise FileNotFoundError(
            f"Template file not found: {INDEX_HTML_PATH}"
        )

    return INDEX_HTML_PATH.read_text(
        encoding="utf-8"
    )


# =========================================================
# REQUEST SCHEMAS
# =========================================================

class SinglePredictionRequest(BaseModel):

    input: Dict[str, Any] = Field(
        ...,
        description="Single customer record for ad_ctr prediction"
    )


class BatchPredictionRequest(BaseModel):

    inputs: List[Dict[str, Any]] = Field(
        ...,
        description="Batch customer records for ad_ctr prediction"
    )


# =========================================================
# RESPONSE SCHEMAS
# =========================================================

class SinglePredictionResponse(BaseModel):

    prediction: int

    probability: Optional[float] = None

    success: bool

    model_path: Optional[str] = None


class BatchPredictionResponse(BaseModel):

    predictions: List[int]

    probabilities: Optional[List[float]] = None

    success: bool

    count: int

    model_path: Optional[str] = None


class HealthResponse(BaseModel):

    status: str

    model_loaded: bool

    model_path: Optional[str] = None

    version: str


# =========================================================
# INITIALIZE INFERENCE
# =========================================================

def initialize_inference() -> ModelInference:

    config = ConfigurationManager()

    return ModelInference(
        config=config.get_model_inference_config()
    )


# =========================================================
# APPLICATION LIFECYCLE
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Downloading model artifacts from S3...")

        download_artifacts()

        logger.info("Initializing inference pipeline...")
        
        app.state.inference = initialize_inference()

        app.state.model_loaded = True

        app.state.model_path = str(
            app.state.inference.config.model_path
        )

        logger.info(
            f"Loaded ad_ctr prediction model from "
            f"{app.state.model_path}"
        )

    except Exception as exc:

        app.state.inference = None

        app.state.model_loaded = False

        app.state.model_path = None

        logger.exception(
            f"Failed to initialize inference model: {exc}"
        )

    yield


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(

    title="AD_CTR Prediction API",

    description=(
        "Production-grade FastAPI service for "
        "ad_ctr prediction using machine learning."
    ),

    version="1.0.0",

    docs_url="/api/docs",

    redoc_url="/api/redoc",

    lifespan=lifespan,
)


# =========================================================
# CORS MIDDLEWARE
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

    allow_credentials=True,
)


# =========================================================
# STATIC FILES
# =========================================================

static_dir = ROOT_DIR / "static"

if static_dir.exists() and static_dir.is_dir():

    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static"
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def read_index() -> HTMLResponse:

    try:

        return HTMLResponse(
            content=load_index_html(),
            status_code=200
        )

    except FileNotFoundError as exc:

        logger.error(str(exc))

        raise HTTPException(
            status_code=500,
            detail="Index page is unavailable"
        )


# =========================================================
# HEALTH CHECKS
# =========================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:

    return HealthResponse(

        status="ok",

        model_loaded=bool(
            getattr(app.state, "model_loaded", False)
        ),

        model_path=getattr(
            app.state,
            "model_path",
            None
        ),

        version=app.version,
    )


@app.get("/health/live")
async def liveness():

    return {
        "status": "alive"
    }


@app.get("/health/ready")
async def readiness():

    if not getattr(app.state, "model_loaded", False):

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "ready"
    }


# =========================================================
# SINGLE PREDICTION
# =========================================================

@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest) -> SinglePredictionResponse:

    if (not getattr(app.state,"model_loaded", False) or app.state.inference is None):

        raise HTTPException(
            status_code=503,
            detail="Inference model is unavailable"
        )

    try:

        df = app.state.inference.predict_records(
            [request.input]
        )

        prediction = int(
            df["prediction"].iloc[0]
        )

        probability = (
            float(
                df["prediction_probability"].iloc[0]
            )
            if "prediction_probability" in df.columns
            else None
        )

        return SinglePredictionResponse(

            prediction=prediction,

            probability=probability,

            success=True,

            model_path=getattr(
                app.state,
                "model_path",
                None
            ),
        )

    except Exception as exc:

        logger.exception(
            f"Single prediction failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# BATCH PREDICTION
# =========================================================

@app.post("/predict/batch",response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:

    if (not getattr(app.state,"model_loaded", False) or app.state.inference is None):

        raise HTTPException(
            status_code=503,
            detail="Inference model is unavailable"
        )

    if not request.inputs:

        raise HTTPException(
            status_code=400,
            detail="At least one record is required"
        )

    try:

        df = app.state.inference.predict_records(
            request.inputs
        )

        predictions = [
            int(value)
            for value in df["prediction"].tolist()
        ]

        probabilities = (df["prediction_probability"].astype(float).tolist()
               if "prediction_probability" in df.columns else None)

        return BatchPredictionResponse(

            predictions=predictions,

            probabilities=probabilities,

            success=True,

            count=len(predictions),

            model_path=getattr(
                app.state,
                "model_path",
                None
            ),
        )

    except Exception as exc:

        logger.exception(
            f"Batch prediction failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# VALIDATION ERROR HANDLER
# =========================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request,exc: RequestValidationError) -> JSONResponse:

    return JSONResponse(

        status_code=422,

        content={
            "errors": exc.errors(),
            "body": exc.body
        },
    )


# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request,exc: Exception) -> JSONResponse:

    logger.exception(
        f"Unhandled server error: {exc}"
    )

    return JSONResponse(

        status_code=500,

        content={
            "detail": "Internal server error"
        },
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(app,host="0.0.0.0",port=8000)