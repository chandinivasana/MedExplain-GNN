# MedExplain-GNN: Explainable Medical Reasoning Engine

**MedExplain-GNN** is an end-to-end, graph-based medical reasoning engine designed to move beyond "black-box" AI. It translates unstructured user symptom descriptions into precise medical predictions and provides contextual dietary precautions based on biological interactions.

## 🚀 Key Features

-   **Explainable Graph Reasoning:** Powered by **Graph Convolutional Networks (GCN)** using **PyTorch Geometric** for disease prediction from symptom node neighborhoods.
-   **Medical NER Pipeline:** Utilizes **BioBERT** for extracting exact symptoms and medical entities from raw text.
-   **Knowledge Graph Traversal:** Built with **Neo4j** to provide transparent, graph-justified reasons for dietary contraindications.
-   **Microservice Architecture:** Distributed system including a **FastAPI** Gateway, **Next.js** Frontend, and dedicated **AI Inference** services.
-   **Production-Ready DevOps:** Containerized with **Docker** and orchestrated via **Kubernetes (K8s)** with automated health probes and persistent logging in **MongoDB**.
-   **Asynchronous Processing:** Integrated with **Redis** for efficient message passing between services.

## 🛠️ Tech Stack

-   **Frontend:** Next.js (TypeScript, Tailwind CSS)
-   **Backend:** FastAPI (Python)
-   **AI Engine:** PyTorch Geometric (GNN), Transformers (BioBERT)
-   **Databases:** Neo4j (Graph), MongoDB (NoSQL Logs), Redis (Cache/Queue)
-   **Infrastructure:** Docker, Kubernetes (Minikube/Kind), Makefile

## 🏗️ Architecture

1.  **Symptom Ingestion:** Users input descriptions into the Next.js interface.
2.  **Entity Extraction:** FastAPI forwards text to the AI Engine where BioBERT extracts symptom entities.
3.  **Graph Inference:** Extracted symptoms map to Neo4j nodes. The GCN performs inference on the local graph structure to predict a disease.
4.  **Explainability:** The system queries Neo4j for nodes connected to the predicted disease via the `[:CONTRAINDICATED]` edge.
5.  **Output:** The user receives a prediction, a detailed explanation, and a list of foods to avoid.

## 🚦 Getting Started

### Prerequisites
- Docker & Docker Compose
- Kubernetes (Minikube or similar)
- Python 3.10+

### Local Development (Docker Compose)
```bash
./setup.sh
```

### Kubernetes Deployment
```bash
# Point shell to minikube docker engine
eval $(minikube docker-env)

# Build images locally
docker build -t dl-frontend:latest ./frontend
docker build -t dl-backend:latest ./backend
docker build -t dl-ai-service:latest ./ai_engine

# Deploy and Populate
make k8s-deploy
make k8s-populate
make k8s-port-forward
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
