# MedExplain-GNN: Explainable Medical Reasoning Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

**MedExplain-GNN** is an end-to-end, graph-based medical reasoning engine. It leverages Graph Attention Networks (GAT) and Knowledge Graphs to translate unstructured symptom descriptions into explainable disease predictions and clinical dietary precautions.

---

## 🏗️ System Architecture

The system is built as a containerized microservices architecture:

1.  **Frontend (Next.js):** Responsive UI for symptom ingestion.
2.  **Gateway (FastAPI):** Orchestrates requests between the UI and AI services.
3.  **AI Engine (FastAPI + PyTorch):** Extracts symptoms via BioBERT and performs GAT inference.
4.  **Knowledge Graph (Neo4j):** Stores relational data for diseases, symptoms, and food contraindications.
5.  **Task Queue (Redis/RQ):** Handles asynchronous inference tasks.
6.  **Persistence (MongoDB):** Logs inference results and system metadata.

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose (v2.0+)

### Quick Launch
The included setup script automates container orchestration, database seeding, and initial model validation.

```bash
chmod +x setup.sh
./setup.sh
```

### Service Dashboard
| Service | URL | Description |
| :--- | :--- | :--- |
| **Frontend** | [http://localhost:3000](http://localhost:3000) | Main User Interface |
| **API Gateway** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger Documentation |
| **AI Service** | [http://localhost:8001](http://localhost:8001) | Inference Engine API |
| **Neo4j** | [http://localhost:7474](http://localhost:7474) | Graph Database Browser |

---

## 🧠 MLOps & Training Pipeline

To maintain or retrain the GAT model within the `ai-service` container:

### 1. Data Ingestion & Graph Seeding
```bash
docker compose exec ai-service python database/seed_database.py
```

### 2. Dataset Reconstruction & Training
```bash
docker compose exec ai-service python dataset_builder.py
docker compose exec ai-service python train.py --epochs 250 --hidden-channels 256
```

### 3. Model Evaluation
```bash
docker compose exec ai-service python evaluate_model.py
```

---

## 🛠️ API Reference

### Disease Prediction
`POST /predict-disease`

**Payload:**
```json
{
  "text": "I have a throbbing headache, nausea, and sensitivity to light."
}
```

**Response:**
```json
{
  "disease": "Migraine",
  "confidence": 0.942,
  "explanation": "Detected: headache, nausea, sensitivity to light. Graph attention weights suggest Migraine.",
  "dietary_precautions": [
    "Avoid: Aged Cheeses",
    "Recommended: Ginger"
  ]
}
```

---

## 📂 Project Structure

```text
├── ai_engine/          # GAT Model, BioBERT NER, and Inference Logic
├── backend/            # FastAPI Gateway and Service Orchestration
├── frontend/           # Next.js UI with Tailwind CSS
├── database/           # Neo4j Seeding and Migration Scripts
├── data/               # Source Datasets and Processed Artifacts
└── k8s/                # Kubernetes Deployment Manifests
```

---

## 🛡️ Disclaimer
*This project is for academic demonstration and research purposes only. It is not intended for clinical use or professional medical diagnosis.*

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
