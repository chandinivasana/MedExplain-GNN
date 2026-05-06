from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import traceback
import uuid
import os
from contextlib import asynccontextmanager
from redis import Redis
from rq import Queue
from motor.motor_asyncio import AsyncIOMotorClient

from symptom_extractor import SymptomExtractor
from inference import get_engine

# --- LIFESPAN FOR MLOPS (LOAD MODEL ONCE) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load GAT Engine into memory
    print("Initializing GAT Inference Engine...")
    try:
        app.state.engine = get_engine()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
        traceback.print_exc()
        app.state.engine = None
    
    app.state.extractor = SymptomExtractor()
    yield
    # Cleanup (if needed)
    print("Shutting down...")

app = FastAPI(title="MedExplain-GNN GAT AI Service", lifespan=lifespan)

# Setup Infrastructure
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/medexplain_logs")

redis_conn = Redis.from_url(REDIS_URL)
inference_queue = Queue('medical_inference', connection=redis_conn)
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client.get_default_database()

class SymptomInput(BaseModel):
    text: str

@app.get("/")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": app.state.engine is not None,
        "queue_size": len(inference_queue)
    }

@app.post("/api/diagnose/async")
async def start_inference(input: SymptomInput):
    """Generates a task ID and pushes to Redis queue for GAT processing."""
    task_id = str(uuid.uuid4())
    # The worker will handle the full pipeline
    inference_queue.enqueue('worker.process_medical_inference', task_id, input.text, job_id=task_id)
    return {"status": "processing", "task_id": task_id}

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """Polls MongoDB for the completed GAT task result."""
    result = await db.inference_results.find_one({"task_id": task_id})
    if not result:
        job = inference_queue.fetch_job(task_id)
        if job:
            return {"status": job.get_status()}
        raise HTTPException(status_code=404, detail="Task ID not found.")
    
    result["_id"] = str(result["_id"])
    return result

@app.post("/process")
async def process_sync(input: SymptomInput):
    """Synchronous GAT inference (Optimized)."""
    if app.state.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        symptoms = app.state.extractor.extract(input.text)
        if not symptoms:
            raise HTTPException(status_code=400, detail="No symptoms detected.")
        
        predictions, dietary_precautions, cypher_query = app.state.engine.predict(symptoms)
        top_disease, top_conf = predictions[0]
        
        return {
            "disease": top_disease,
            "confidence": float(top_conf),
            "explanation": f"Detected symptoms: {', '.join(symptoms)}. The GAT model identified {top_disease} as the most likely diagnosis based on graph attention weights.",
            "predictions": [{"disease": d, "confidence": float(c)} for d, c in predictions],
            "dietary_precautions": dietary_precautions,
            "cypher_query": cypher_query
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
