# Deployment Guide: MedExplain-GNN

This guide explains how to deploy **MedExplain-GNN** across cloud platforms for live accessibility, interactive reviewer verification, and production use.

---

## 1. Zero-Cost Live Demo: Hugging Face Spaces (Recommended for Portfolio)

Hugging Face Spaces allows you to deploy the interactive Gradio demo (`demo_app.py`) for **free**, providing a public HTTPS URL accessible to recruiters and evaluators worldwide.

### Steps:
1. Log in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Fill in the details:
   - **Space name**: `MedExplain-GNN`
   - **License**: `MIT`
   - **SDK**: `Gradio`
   - **Hardware**: `CPU basic · 2 vCPU · 16GB · Free`
3. Clone your Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/MedExplain-GNN hf-space
   ```
4. Copy the required application files into the Space repository:
   ```bash
   cp demo_app.py hf-space/app.py
   cp -r ai_engine hf-space/
   cp -r data hf-space/
   ```
5. Create a `requirements.txt` in the Space root:
   ```text
   torch>=2.2.0
   torch-geometric>=2.4.0
   transformers>=4.38.0
   gradio>=4.44.0
   neo4j>=5.17.0
   pandas>=2.0.0
   numpy>=1.24.0
   ```
6. Commit and push:
   ```bash
   cd hf-space
   git add .
   git commit -m "feat: deploy MedExplain-GNN interactive demo"
   git push
   ```
7. Hugging Face will automatically build and launch your demo at:
   `https://huggingface.co/spaces/<your-username>/MedExplain-GNN`

---

## 2. Next.js Frontend Deployment: Vercel (1-Click)

The Next.js 16 frontend is configured for deployment on [Vercel](https://vercel.com).

### Steps:
1. Go to [Vercel](https://vercel.com) and click **Add New Project**.
2. Import your GitHub repository: `chandinivasana/MedExplain-GNN`.
3. Set the **Root Directory** to `frontend`.
4. Configure Environment Variables:
   - `NEXT_PUBLIC_API_URL`: Your deployed backend API URL (e.g., `https://medexplain-api.onrender.com` or leave blank to engage the interactive demo preview).
5. Click **Deploy**.
6. Your frontend will be live at `https://medexplain-gnn.vercel.app`.

---

## 3. Full-Stack Local Deployment: Docker Compose

To run all 6 microservices locally in isolated containers:

```bash
# Clone the repository
git clone https://github.com/chandinivasana/MedExplain-GNN.git
cd MedExplain-GNN

# Copy environment variables
cp .env.example .env

# Build and start all containers
docker compose up --build -d
```

### Verified Endpoints:
| Service | URL | Default Credentials |
| --- | --- | --- |
| **Next.js UI** | `http://localhost:3000` | - |
| **FastAPI Gateway** | `http://localhost:8000/docs` | Swagger UI |
| **AI Inference Service** | `http://localhost:8001/docs` | OpenAPI UI |
| **Neo4j Browser** | `http://localhost:7474` | `neo4j` / `password` |
| **MongoDB** | `mongodb://localhost:27017` | `medexplain_logs` |
| **Redis** | `redis://localhost:6379/0` | - |

---

## 4. Kubernetes (k8s) Production Cluster Deployment

For production Kubernetes environments (EKS, GKE, Minikube):

```bash
# Apply namespace, ConfigMaps, and StatefulSets
make k8s-deploy

# Check cluster rollout status
make k8s-status

# Forward ports to local workstation
make k8s-port-forward
```
