# MedExplain-GNN: Explainable Medical Reasoning Engine

MedExplain-GNN is a sophisticated, graph-based medical reasoning system that integrates Natural Language Processing (NLP) with Graph Neural Networks (GNN) to provide transparent and explainable disease predictions. By combining BioBERT embeddings with Graph Attention Networks (GAT), the system moves beyond black-box AI to offer clinically-grounded justifications and dietary precautions based on Knowledge Graph traversal.

## Core Technologies

- **Graph Attention Networks (GAT):** Implements multi-head attention mechanisms to dynamically weight symptoms based on their clinical significance within the graph topology.
- **BioBERT Embeddings:** Utilizes 768-dimensional biomedical language embeddings for node initialization, ensuring the model understands the semantic context of medical terms.
- **Neo4j Knowledge Graph:** Stores complex relationships between diseases, symptoms, and dietary contraindications for real-time explainable reasoning.
- **MLOps Pipeline:** Features a decoupled training-to-inference workflow with model persistence (.pth) and automated data cleaning for production readiness.
- **Full-Stack Integration:** A modern architecture featuring a Next.js (TypeScript) frontend, a FastAPI gateway, and a specialized AI inference service.

## Architecture

1. **Entity Extraction:** Raw patient descriptions are processed via BioBERT NER to identify medical symptoms.
2. **Graph Inference:** Identified symptoms activate nodes in a GAT model trained on a curated medical dataset.
3. **Calibration:** Temperature scaling is applied to the output logits to provide well-calibrated confidence scores.
4. **Explainability:** The system queries Neo4j for associated dietary risks and generates Cypher queries to visualize the reasoning path.

## Project Structure

- `ai_engine/`: Core AI services including the GAT model, training scripts, and inference engine.
- `backend/`: FastAPI Gateway that manages request routing and logging.
- `frontend/`: Next.js web application for user interaction and results visualization.
- `data/`: Curated medical datasets and automated cleaning scripts.
- `database/`: Neo4j seeding and graph population scripts.

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (optional for full stack deployment)

### Local Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/chandinivasana/MedExplain-GNN.git
cd MedExplain-GNN
```

2. **Setup the Python environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai_engine/requirements.txt
```

3. **Install Frontend dependencies:**
```bash
cd frontend
npm install
cd ..
```

### MLOps Pipeline: Data & Training

To prepare the model for use, follow the data processing and training sequence:

```bash
# 1. Clean the raw medical dataset
python3 clean_data.py

# 2. Build the graph dataset with BioBERT embeddings
python3 ai_engine/dataset_builder.py

# 3. Train the GAT model (supports MPS/CUDA acceleration)
python3 ai_engine/train.py
```

### Running the Full Stack

For local testing without full infrastructure, use the simulated server mode:

1. **Start the AI Inference Service:**
```bash
python3 mock_server.py
```

2. **Start the Backend Gateway:**
```bash
python3 mock_backend.py
```

3. **Start the Frontend UI:**
```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to interact with the system.

## Testing and Validation

Comprehensive validation scripts are included to verify the integration of the BioBERT-GAT pipeline:

```bash
# Run end-to-end GAT inference demo
python3 run_gat_demo.py
```

## Deployment

The project is ready for containerized deployment using Docker:
```bash
docker-compose up --build
```
Or orchestrated via Kubernetes using the manifests provided in the `k8s/` directory.
