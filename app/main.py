from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.predictor.ml_predictor import DarkPatternPredictor, get_predictor
from app.schemas.schemas import (
    GenericDetectInstance,
    GenericDetectRequest,
    GenericDetectResponse,
    ScarcityInstance,
    ScarcityRequest,
    ScarcityResponse,
    UrgencyInstance,
    UrgencyRequest,
    UrgencyResponse,
)

import structlog
logger = structlog.get_logger()

# This will hold our global predictor instance
predictor_instance: DarkPatternPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor_instance
    logger.info("Pre-loading ML model into memory...")
    try:
        predictor_instance = get_predictor()
        logger.info("Model loaded successfully!")
    except Exception:
        logger.exception("Failed to pre-load model")
    yield
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "model_loaded": predictor_instance is not None
    }


def _get_predictions(texts: list[str]):
    if not predictor_instance:
        raise RuntimeError("Predictor model is not loaded.")
    return predictor_instance.predict(texts)


@app.post("/scarcity", response_model=ScarcityResponse)
def detect_scarcity(request: ScarcityRequest):
    try:
        raw_texts = [item.text for item in request.texts]
        if not raw_texts:
            return ScarcityResponse(version="1.0", instances=[])

        predictions = _get_predictions(raw_texts)

        instances = []
        for i, item in enumerate(request.texts):
            pred = predictions[i]
            has_scarcity = "fake_scarcity" in pred["labels"]
            instances.append(
                ScarcityInstance(
                    text=item.text,
                    path=item.path,
                    id=item.id,
                    has_scarcity=has_scarcity,
                )
            )

        return ScarcityResponse(version="1.0", instances=instances)
    except Exception as e:
        logger.exception("Error processing scarcity request")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/urgency", response_model=UrgencyResponse)
def detect_urgency(request: UrgencyRequest):
    try:
        raw_texts = [item.text for item in request.texts]
        if not raw_texts:
            return UrgencyResponse(version="1.0", urgency_instances=[])

        predictions = _get_predictions(raw_texts)

        instances = []
        for i, item in enumerate(request.texts):
            pred = predictions[i]
            has_urgency = "fake_urgency" in pred["labels"]
            instances.append(
                UrgencyInstance(
                    text=item.text, path=item.path, id=item.id, has_urgency=has_urgency
                )
            )

        return UrgencyResponse(version="1.0", urgency_instances=instances)
    except Exception as e:
        logger.exception("Error processing urgency request")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shaming")
async def detect_shaming(request: Request):
    try:
        json_data = await request.json()

        if json_data.get("Version", "0.1") != "0.2":
            tokens = json_data.get("tokens", [])
            raw_texts = [t.get("text", "") for t in tokens]

            sentences = []
            if raw_texts:
                predictions = _get_predictions(raw_texts)
                for i, token in enumerate(tokens):
                    if "shaming" in predictions[i]["labels"]:
                        sentences.append(
                            {
                                "text": token.get("text", ""),
                                "path": token.get("path", ""),
                                "id": token.get("id", ""),
                            }
                        )
            return sentences
        else:
            texts = json_data.get("texts", [])
            raw_texts = [t.get("text", "") for t in texts]

            instances = []
            if raw_texts:
                predictions = _get_predictions(raw_texts)
                for i, t in enumerate(texts):
                    instances.append(
                        {
                            "text": t.get("text", ""),
                            "path": t.get("path", ""),
                            "id": t.get("id", ""),
                            "has_shaming": "shaming" in predictions[i]["labels"],
                        }
                    )
            return {"version": "0.2", "instances": instances}

    except Exception as e:
        logger.exception("Error processing shaming request")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=GenericDetectResponse)
def detect_dark_patterns(request: GenericDetectRequest):
    try:
        texts_info = request.texts if request.texts else request.tokens
        if not texts_info:
            return GenericDetectResponse(version="1.0", instances=[])

        raw_texts = [item.text for item in texts_info]
        predictions = _get_predictions(raw_texts)

        instances = []
        for i, item in enumerate(texts_info):
            pred = predictions[i]
            instances.append(
                GenericDetectInstance(
                    text=item.text,
                    path=item.path,
                    id=item.id,
                    detected=pred["detected"],
                    labels=pred["labels"],
                )
            )

        return GenericDetectResponse(version="1.0", instances=instances)
    except Exception as e:
        logger.exception("Error processing detect request")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
