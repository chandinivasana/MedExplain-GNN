from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from bson import ObjectId

app = FastAPI(title="MedExplain-GNN Gateway")

# Allow CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8001")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/medexplain_logs")

client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=2000)
db = client.get_default_database()
logs_collection = db.get_collection("prediction_logs")

class PredictionInput(BaseModel):
    text: str

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MedExplain-GNN Gateway",
        "ai_service_url": AI_SERVICE_URL,
        "version": "1.0.0"
    }

@app.post("/predict-disease")
async def forward_prediction(input: PredictionInput):
    clean_text = input.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Symptom text cannot be empty.")
    if len(clean_text) > 5000:
        raise HTTPException(status_code=400, detail="Input text exceeds maximum length of 5000 characters.")

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            response = await http_client.post(f"{AI_SERVICE_URL}/process", json={"text": clean_text})
            response.raise_for_status()
            result = response.json()
            
            # Log session to MongoDB with graceful degradation
            try:
                log_entry = {
                    "input_text": clean_text,
                    "prediction": result,
                    "timestamp": datetime.utcnow()
                }
                await logs_collection.insert_one(log_entry)
            except Exception as mongo_err:
                print(f"Notice: Session logging skipped (MongoDB offline: {mongo_err})")
            
            return result
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Error connecting to AI service: {str(exc)}")

@app.get("/history")
async def get_history():
    try:
        cursor = logs_collection.find({}).sort("timestamp", -1).limit(10)
        logs = await cursor.to_list(length=10)
        for log in logs:
            log["_id"] = str(log["_id"])
        return logs
    except Exception as exc:
        print(f"Notice: Failed to fetch history: {exc}")
        return []

@app.delete("/history/{log_id}")
async def delete_history_item(log_id: str):
    try:
        result = await logs_collection.delete_one({"_id": ObjectId(log_id)})
        if result.deleted_count == 1:
            return {"message": "Log deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Log not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting history: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
