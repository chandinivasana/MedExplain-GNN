from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
from unittest.mock import MagicMock

app = FastAPI(title="MedExplain-GNN Mock Gateway")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_SERVICE_URL = "http://127.0.0.1:8001"

class PredictionInput(BaseModel):
    text: str

@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict-disease")
async def forward_prediction(input: PredictionInput):
    async with httpx.AsyncClient() as http_client:
        try:
            # Connect to the mock AI service
            response = await http_client.post(f"{AI_SERVICE_URL}/process", json={"text": input.text})
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"Error connecting to AI service: {exc}")
            # Fallback mock response if AI service is down or error occurs
            return {
                "disease": "System Error (Mock Fallback)",
                "confidence": 0.0,
                "predictions": [],
                "dietary_precautions": ["Consult a doctor"],
                "cypher_query": "N/A"
            }

@app.get("/history")
async def get_history():
    return []

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
