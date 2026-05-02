# MedExplain-GNN: Explainable Medical Reasoning Engine

MedExplain-GNN is a graph-based medical reasoning engine that translates unstructured symptom descriptions into precise disease predictions and contextual dietary precautions. The system utilizes Graph Attention Networks (GAT) and BioBERT embeddings to provide transparent, clinically-grounded reasoning.

## Key Features

- **Graph Attention Networks (GAT):** Uses GATConv layers to dynamically weight symptoms based on clinical importance.
- **BioBERT Node Embeddings:** Initializes graph nodes with 768-dimensional medical embeddings for semantic reasoning.
- **Explainable Reasoning:** Queries a Neo4j knowledge graph to justify predictions and provide dietary contraindications.
- **MLOps Pipeline:** Decoupled training and inference with model persistence (.pth) and temperature-calibrated outputs.
- **Microservice Architecture:** Distributed system including FastAPI Gateway, Next.js Frontend, and Redis/RQ asynchronous processing.

## Technical Architecture

1. **Extraction:** Raw text is processed via BioBERT NER to identify medical entities.
2. **Embedding:** Identified symptoms are mapped to nodes initialized with BioBERT embeddings.
3. **Inference:** A trained GAT model performs node classification across the symptom-disease graph.
4. **Calibration:** Temperature scaling is applied to logits to ensure realistic confidence scoring.
5. **Explainability:** Neo4j is queried for [:CONTRAINDICATED] and [:RECOMMENDED] edges related to the predicted disease.

## Project Structure

- `ai_engine/model.py`: PyTorch Geometric GAT architecture.
- `ai_engine/dataset_builder.py`: BioBERT embedding generation and graph construction.
- `ai_engine/train.py`: Offline training script with early stopping.
- `ai_engine/inference.py`: Model loading and calibrated prediction logic.
- `ai_engine/main.py`: FastAPI service with lifespan model management.
- `ai_engine/worker.py`: Asynchronous task worker for background inference.

## Installation and Setup

### Prerequisites
- Python 3.10+
- Neo4j Database
- Redis (for async tasks)
- MongoDB (for logging)

### Local Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai_engine/requirements.txt
```

### Model Training (MLOps Pipeline)
```bash
# 1. Build the graph dataset with BioBERT embeddings
python3 ai_engine/dataset_builder.py

# 2. Train the GAT model
python3 ai_engine/train.py
```

### Running the Services
```bash
# Start the AI Service
uvicorn ai_engine.main:app --host 0.0.0.0 --port 8001
```

## Testing

A demonstration script is provided to verify the full pipeline (Extraction -> GAT Inference -> Explainability):
```bash
python3 run_gat_demo.py
```
