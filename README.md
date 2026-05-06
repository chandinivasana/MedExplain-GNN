# MedExplain-GNN: Explainable Medical Reasoning Engine

**MedExplain-GNN** is an end-to-end, graph-based medical reasoning engine. It translates unstructured symptom descriptions into disease predictions and returns contextual dietary precautions through a full-stack, containerized application.

> This project is intended for academic demonstration and engineering evaluation. It is not a clinical diagnostic system.

## Key Features

- **Graph Attention Reasoning:** PyTorch Geometric `MedicalGAT` model for disease prediction from symptom graph signals.
- **Imbalance-Aware Training:** Focal Loss, class weighting, validation-mask training, and validation-loss early stopping.
- **Calibrated Inference:** Temperature-scaled softmax with bounded symptom boosting to avoid extreme out-of-distribution feature shifts.
- **Model QA:** `evaluate_model.py` reports accuracy, macro F1, per-class recall, confusion matrix, rare-class examples, and top-3 predictions.
- **Knowledge Graph Traversal:** Neo4j stores symptom, disease, and dietary precaution relationships.
- **Full-Stack App:** Next.js frontend, FastAPI backend gateway, dedicated FastAPI AI service, MongoDB logging, Redis/RQ, and Docker Compose.

## Tech Stack

- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Backend Gateway:** FastAPI
- **AI Service:** FastAPI, PyTorch, PyTorch Geometric, RQ worker support
- **Databases:** Neo4j, MongoDB
- **Infrastructure:** Docker Compose, Kubernetes manifests, Redis

## Architecture

1. **Symptom Ingestion:** The user enters a free-text symptom description in the Next.js UI.
2. **Gateway Routing:** The frontend sends the request to the backend gateway at `localhost:8000`.
3. **Symptom Extraction:** The backend forwards text to the AI service at `localhost:8001`.
4. **GAT Inference:** Extracted symptoms are mapped to graph node indices and passed through the trained `MedicalGAT`.
5. **Response:** The user receives the predicted disease, confidence, explanation, and dietary precautions.

## Local Development

### Prerequisites

- Docker & Docker Compose
- Python is not required on the host when running through Docker

### Start the Full Stack

```bash
docker compose up --build -d
```

Services:

```text
Frontend:    http://localhost:3000
Backend:     http://localhost:8000
AI Service:  http://localhost:8001
Neo4j:       http://localhost:7474
MongoDB:     localhost:27017
Redis:       localhost:6379
```

### Generate Graph Data and Train the GAT

Run these inside the AI service container:

```bash
docker compose exec ai-service python dataset_builder.py
docker compose exec ai-service python train.py \
  --data-path graph_data.pt \
  --save-path best_model.pth \
  --hidden-channels 256 \
  --epochs 250 \
  --patience 50 \
  --gamma 2.0
```

Expected model artifacts in `/app`:

```text
best_model.pth
graph_data.pt
index_to_disease.json
```

Restart the AI service after training:

```bash
docker compose restart ai-service
```

### Evaluate the Model

```bash
docker compose exec ai-service python evaluate_model.py \
  --graph-data-path graph_data.pt \
  --checkpoint-path best_model.pth
```

The evaluator writes:

```text
evaluation_report/evaluation_report.json
evaluation_report/confusion_matrix.csv
```

The report includes accuracy, macro F1, per-class precision/recall/F1, rare-class accuracy, top-3 predictions, saved validation loss, and confidence calibration.

### Verify Real Model Inference

Check artifacts:

```bash
docker compose exec ai-service ls -lh best_model.pth graph_data.pt index_to_disease.json
```

Check that the service is not falling back to demo predictions:

```bash
docker compose logs --tail=80 ai-service
```

Run a live request:

```bash
curl -X POST http://localhost:8000/predict-disease \
  -H "Content-Type: application/json" \
  -d '{"text":"I have high fever, severe joint pain, and pain behind my eyes."}'
```

Expected example response after training:

```json
{
  "disease": "Dengue",
  "confidence": 0.981683611869812,
  "explanation": "Detected symptoms: fever, joint pain, pain behind eyes. The GAT model identified Dengue as the most likely diagnosis based on graph attention weights.",
  "predictions": [
    {"disease": "Dengue", "confidence": 0.981683611869812}
  ],
  "dietary_precautions": ["Avoid: Salty Foods"]
}
```

## Notes on Model Quality

The included `dataset_builder.py` creates a compact demonstration dataset so the full MLOps path can run locally end to end. It is useful for final-year project demonstration, Dockerized model serving, and QA validation.

For clinical-grade claims, replace the generated dataset with a validated medical dataset and report held-out performance, macro F1, per-class recall, confusion matrix, calibration, and rare-disease behavior.

## Kubernetes Deployment

```bash
eval $(minikube docker-env)

docker build -t dl-frontend:latest ./frontend
docker build -t dl-backend:latest ./backend
docker build -t dl-ai-service:latest ./ai_engine

make k8s-deploy
make k8s-populate
make k8s-port-forward
```
