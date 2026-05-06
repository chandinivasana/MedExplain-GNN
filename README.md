# MedExplain-GNN: Explainable Medical Reasoning Engine

**MedExplain-GNN** is a graph-based medical reasoning engine that translates symptom descriptions into disease predictions with contextual dietary precautions using a Graph Attention Network (GAT).

## Quick Start (Docker)

Ensure you have Docker and Docker Compose installed.

### 1. Start the System
```bash
./setup.sh
```

### 2. Services
- **Frontend:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **AI Service:** [http://localhost:8001](http://localhost:8001)
- **Neo4j Console:** [http://localhost:7474](http://localhost:7474) (user: `neo4j`, pass: `password`)

---

## MLOps: Training & Evaluation

To retrain the GAT model on the latest graph data:

### 1. Rebuild Dataset & Train
```bash
docker compose exec ai-service python dataset_builder.py
docker compose exec ai-service python train.py --epochs 250
```

### 2. Evaluate Performance
```bash
docker compose exec ai-service python evaluate_model.py
```

### 3. Reload Model
```bash
docker compose restart ai-service
```

---

## API Usage

### Predict Disease
```bash
curl -X POST http://localhost:8000/predict-disease \
  -H "Content-Type: application/json" \
  -d '{"text":"I have high fever, severe joint pain, and pain behind my eyes."}'
```

**Example Response:**
```json
{
  "disease": "Dengue",
  "confidence": 0.98,
  "explanation": "Detected symptoms: fever, joint pain, pain behind eyes...",
  "dietary_precautions": ["Avoid: Salty Foods"]
}
```

---

## Technical Stack
- **AI:** PyTorch Geometric (GAT), BioBERT (NER)
- **Backend:** FastAPI, Redis/RQ, MongoDB
- **Graph:** Neo4j
- **Frontend:** Next.js, Tailwind CSS
