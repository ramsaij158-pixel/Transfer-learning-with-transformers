# Interview Guide: Production ML Project Classifier

## 30-second project summary

“This project is an end-to-end NLP system that classifies machine-learning project descriptions into tags. Raw CSV data is validated, split with class stratification, cleaned and tokenized with SciBERT. A PyTorch classification head is trained using Ray Train, experiments and artifacts are tracked in MLflow, and the best checkpoint is evaluated before it is served through a validated FastAPI endpoint on Ray Serve. CI runs quality and data/model tests, while deployment configuration supports batch jobs and horizontally scalable online inference.”

## The business problem

Users submit a project title and description. The service assigns a category such as NLP, computer vision, MLOps, or “other.” In a marketplace or internal knowledge catalog, this reduces manual tagging and improves search, routing, and analytics. A confidence threshold sends uncertain cases to “other,” which can become a human-review queue in a real product.

## End-to-end workflow

1. `datasets/*.csv` contains the raw projects, labels, and a holdout set.
2. `madewithml/data.py` loads data with Ray Data, creates reproducible stratified train/validation splits, cleans text, maps labels to stable integer IDs, and tokenizes text with SciBERT.
3. `madewithml/models.py` wraps the pretrained SciBERT encoder with dropout and a linear multiclass classification head. It also owns checkpoint save/load behavior.
4. `madewithml/train.py` builds distributed Ray workers, runs forward/backward passes, evaluates each epoch, changes the learning rate when validation loss plateaus, and retains the best checkpoint.
5. `madewithml/tune.py` searches hyperparameters rather than selecting them manually.
6. MLflow records parameters, metrics, artifacts, and run IDs so every model is reproducible and auditable.
7. `madewithml/evaluate.py` loads a checkpoint and calculates offline metrics on unseen data. Model tests add behavioral checks beyond one aggregate score.
8. `madewithml/predict.py` resolves the selected MLflow run, restores the preprocessor and model together, runs inference, and maps numeric outputs back to business labels.
9. `madewithml/serve.py` exposes the model using FastAPI and Ray Serve. It validates payload sizes, returns model/run lineage and latency, logs a request ID, provides health/readiness endpoints, and rejects low-confidence output as “other.”
10. `.github/workflows/` runs automated quality/testing and workload pipelines. `deploy/` defines remote Ray jobs and the serving application. `Dockerfile` provides a reproducible, non-root runtime.

## File-by-file explanation

| File or directory | Responsibility | Interview point |
|---|---|---|
| `madewithml/config.py` | Paths, MLflow URI, logging, constants | Configuration can be overridden by environment instead of changing code. |
| `madewithml/data.py` | Ingestion, splitting, cleaning, tokenization | The same fitted preprocessor is reused at inference to prevent training-serving skew. |
| `madewithml/models.py` | Neural-network architecture and serialization | The checkpoint stores architecture arguments plus learned weights. |
| `madewithml/train.py` | Distributed training and checkpoint selection | Ray handles scaling; validation loss determines which checkpoint survives. |
| `madewithml/tune.py` | Hyperparameter search | Tuning is a separate workload and does not complicate the core training loop. |
| `madewithml/evaluate.py` | Offline quality assessment | Evaluation uses holdout data and class-level metrics, not training accuracy. |
| `madewithml/predict.py` | Model restoration and batch prediction | It couples label mapping, preprocessing, and the model into one predictor. |
| `madewithml/serve.py` | Online API and confidence policy | The API contract, observability fields, readiness, and fallback behavior are production concerns. |
| `tests/code` | Unit and integration-style code tests | Fast feedback for deterministic logic and API contracts. |
| `tests/data` | Dataset expectation tests | Data is an external dependency and must be validated like code. |
| `tests/model` | Behavioral model tests | Checks important slices and invariants that aggregate metrics can hide. |
| `deploy` | Ray job/service infrastructure configuration | Training and serving are repeatable workloads rather than laptop commands. |
| `.github/workflows` | CI and controlled workload automation | Least-privilege permissions and checks protect the main branch. |

## Design decisions to explain

### Why SciBERT?

The input is technical ML-project text. SciBERT was pretrained on scientific language, so its vocabulary and representations are a reasonable transfer-learning baseline. The project fine-tunes it instead of training a transformer from scratch, reducing data and compute requirements.

### Why Ray?

Ray provides one Python abstraction across scalable data processing, distributed training, tuning, batch inference, and serving. The same application can run locally and later use multiple workers or replicas.

### Why MLflow?

MLflow creates lineage between source configuration, hyperparameters, metrics, and model artifacts. A production request returns the model run ID, which helps trace incidents back to the exact model version.

### Why an “other” threshold?

The argmax class is always returned even when every class probability is weak. The threshold turns uncertainty into an explicit product policy. In production, the threshold should be selected using validation data and business costs, then monitored for coverage and error rate.

## Production improvements in this edition

- API request schemas enforce required text and safe size limits.
- Responses expose request ID, model run ID, and latency for troubleshooting.
- Low-confidence routing is isolated and unit-tested.
- Readiness is separate from the basic liveness endpoint.
- MLflow and serving resources are environment-configurable.
- Label-to-index construction is deterministic, which prevents label drift between runs.
- URL cleanup happens before punctuation processing, fixing a preprocessing bug.
- Container execution uses a non-root user and excludes development artifacts.
- CI uses explicit least-privilege permissions and current action versions.

## Honest limitations and next steps

Say these proactively; they show engineering judgment:

- The dataset is small and may not represent production traffic. Add slice analysis, drift monitoring, and a feedback loop.
- File-based MLflow storage is appropriate locally but should become a managed/shared tracking server plus object storage in production.
- Authentication, rate limiting, TLS, and network policy belong at an API gateway or platform layer.
- Dependencies reflect the original Ray 2.7/Python 3.10 stack. A separate migration should upgrade Ray, Transformers, PyTorch, and FastAPI together, followed by full regression and load testing.
- The `/evaluate` route is useful for controlled operations but should not be publicly exposed because it launches a costly offline workload.
- Add image signing, vulnerability scanning, SBOM generation, secrets management, autoscaling, canary rollout, and rollback policy before a real launch.

## Common interview questions

**How do you prevent data leakage?**  
The split occurs before the preprocessor is fitted. Label mappings are learned from training data, and the holdout set is reserved for final evaluation.

**How do you prevent training-serving skew?**  
The fitted preprocessor metadata and model checkpoint are restored together, and both offline and online prediction call the same preprocessing code.

**How would you monitor it?**  
Track latency, request/error rate, resource use, confidence distribution, “other” rate, input drift, class distribution, and delayed ground-truth metrics. Join those signals to `model_run_id` and request ID.

**How would you deploy a new model safely?**  
Require data/code/model tests, register an immutable candidate, deploy it as a canary or shadow, compare service and quality metrics, gradually shift traffic, and retain the previous run ID for rollback.

**What happens when confidence is low?**  
The service returns “other.” In a mature system, that event is queued for human labeling, later added to a reviewed dataset, and used in the next training cycle.

**What would you optimize first?**  
Measure first. Likely options are dynamic batching, smaller/distilled models, quantization, tokenizer caching, GPU replicas, or autoscaling. The right choice depends on whether latency, throughput, or cost is the constraint.

## Two-minute interview narrative

“I treated this as a product system rather than only a notebook model. Data enters through a reproducible Ray Data pipeline, where I make a stratified split and apply one shared preprocessing implementation. SciBERT produces domain-aware embeddings, and a lightweight PyTorch head predicts the final class. Ray Train lets the same training loop scale to multiple workers, while MLflow preserves parameters, metrics, artifacts, and the model run ID. I select checkpoints using validation loss, then run offline and behavioral tests on unseen data.

For inference, the predictor restores both the exact label mapping and model weights, avoiding training-serving skew. Ray Serve hosts a FastAPI contract with input validation, readiness, request correlation, lineage, and latency. A confidence threshold routes uncertain results to ‘other’ instead of pretending every prediction is trustworthy. CI validates code, data, and model behavior, and the Docker image runs as a non-root user. For a real launch, my next steps would be managed artifact storage, gateway security, monitoring and drift alerts, canary deployment, and a human feedback loop.”
