import argparse
import logging
import os
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

import ray
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ray import serve
from starlette.requests import Request

from madewithml import evaluate, predict
from madewithml.config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(
    title="Made With ML",
    description="Classify machine learning projects.",
    version="1.0.0",
)

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Validated online inference contract."""

    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., min_length=1, max_length=5000)


class Prediction(BaseModel):
    prediction: str
    probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    request_id: str
    model_run_id: str
    latency_ms: float
    results: List[Prediction]


def apply_confidence_threshold(results: List[Dict], threshold: float) -> List[Dict]:
    """Route low-confidence predictions to the human-review/other class."""
    for result in results:
        prediction = result["prediction"]
        if result["probabilities"][prediction] < threshold:
            result["prediction"] = "other"
    return results


@serve.deployment(
    num_replicas=int(os.getenv("MODEL_REPLICAS", "1")),
    ray_actor_options={
        "num_cpus": float(os.getenv("MODEL_CPUS", "1")),
        "num_gpus": float(os.getenv("MODEL_GPUS", "0")),
    },
)
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: float = 0.9):
        """Initialize the model."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/app", include_in_schema=False)
    def _web_app(self):
        """Serve the browser UI for the live classifier."""
        return FileResponse(Path(__file__).parent / "web" / "index.html")

    @app.get("/health/ready")
    def _readiness(self) -> Dict:
        """Readiness probe: the deployment is ready after the predictor loads."""
        return {"status": "ready", "model_run_id": self.run_id}

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/evaluate/")
    async def _evaluate(self, request: Request) -> Dict:
        data = await request.json()
        results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @app.post("/predict/", response_model=PredictionResponse)
    async def _predict(self, payload: PredictionRequest) -> Dict:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            sample_ds = ray.data.from_items(
                [{"title": payload.title, "description": payload.description, "tag": "other"}]
            )
            results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)
            results = apply_confidence_threshold(results, self.threshold)
        except Exception as exc:
            logger.exception("prediction_failed request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="prediction failed") from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("prediction_complete request_id=%s latency_ms=%s", request_id, latency_ms)
        return {
            "request_id": request_id,
            "model_run_id": self.run_id,
            "latency_ms": latency_ms,
            "results": results,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    args = parser.parse_args()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))
