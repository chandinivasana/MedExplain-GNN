import os
import time
import uuid
import asyncio
from redis import Redis
from rq import Worker, Queue, Connection
from motor.motor_asyncio import AsyncIOMotorClient

# New GAT Inference Engine
from symptom_extractor import SymptomExtractor
from inference import get_engine

# Environment Config
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/medexplain_logs")

# Pre-load extractor and engine
extractor = SymptomExtractor()
# Engine will be initialized on first call or we can trigger it here
try:
    engine = get_engine()
except Exception as e:
    print(f"Warning: Engine initialization failed: {e}")
    engine = None

def process_medical_inference(task_id, text):
    """
    Background Task: Runs full BioBERT -> GAT -> Neo4j Pipeline
    """
    print(f"[Worker] Starting GAT inference for task {task_id}")
    
    # 1. Extraction (BioBERT NER)
    symptoms = extractor.extract(text)
    
    # 2. GAT Inference
    if not symptoms:
        result = {
            "task_id": task_id,
            "status": "failed",
            "error": "No symptoms detected",
            "timestamp": time.time()
        }
    elif engine is None:
        result = {
            "task_id": task_id,
            "status": "failed",
            "error": "Inference engine not initialized",
            "timestamp": time.time()
        }
    else:
        predictions, dietary_precautions, cypher_query = engine.predict(symptoms)
        top_disease, top_conf = predictions[0]
        
        result = {
            "task_id": task_id,
            "status": "completed",
            "disease": top_disease,
            "confidence": float(top_conf),
            "explanation": f"Detected: {', '.join(symptoms)}. GAT Predictions: {', '.join([f'{d} ({c*100:.1f}%)' for d, c in predictions])}",
            "dietary_precautions": dietary_precautions,
            "cypher_query": cypher_query,
            "timestamp": time.time()
        }

    # 3. Persistence (MongoDB)
    async def save_to_mongo():
        client = AsyncIOMotorClient(MONGO_URI)
        db = client.get_default_database()
        await db.inference_results.update_one(
            {"task_id": task_id},
            {"$set": result},
            upsert=True
        )
        print(f"[Worker] Task {task_id} saved to MongoDB")

    # Run the async save
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(save_to_mongo())
    else:
        loop.run_until_complete(save_to_mongo())
    
    return result

if __name__ == '__main__':
    redis_conn = Redis.from_url(REDIS_URL)
    with Connection(redis_conn):
        worker = Worker(['medical_inference'])
        print("[Worker] GAT Inference Worker listening...")
        worker.work()
