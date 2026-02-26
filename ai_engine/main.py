from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from symptom_extractor import SymptomExtractor
from gcn_model import get_disease_prediction

app = FastAPI(title="MedExplain-GNN AI Service")
extractor = SymptomExtractor()

class SymptomInput(BaseModel):
    text: str

class PredictionResult(BaseModel):
    disease: str
    confidence: float
    explanation: str
    dietary_precautions: List[str]

@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/process")
async def process_request(input: SymptomInput):
    symptoms = extractor.extract(input.text)
    if not symptoms:
        raise HTTPException(status_code=400, detail="No symptoms detected.")
    
    disease, confidence = get_disease_prediction(symptoms)
    # In a real app, this would query Neo4j for dietary_precautions
    # Mocking for logic flow:
    dietary_precautions = ["Avoid sugary foods"] if "Influenza" in disease else ["Avoid salty foods"]
    
    return PredictionResult(
        disease=disease,
        confidence=confidence,
        explanation=f"Based on detected symptoms: {', '.join(symptoms)}.",
        dietary_precautions=dietary_precautions
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
