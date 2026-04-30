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
    cypher_query: str

@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/process")
async def process_request(input: SymptomInput):
    # 1. Advanced NER Extraction
    symptoms = extractor.extract(input.text)
    if not symptoms:
        raise HTTPException(status_code=400, detail="No recognized medical symptoms detected in the text.")
    
    # 2. PyTorch Geometric Inference
    top_disease, top_conf, all_predictions, dietary_precautions = get_disease_prediction(symptoms)
    
    # 3. Build Explainability String
    explanation = f"Detected symptoms via BioBERT: {', '.join(symptoms)}. GNN Top Matches: " + ", ".join([f"{d} ({c*100:.1f}%)" for d, c in all_predictions])
    
    # Cypher query used for dietary precautions
    cypher_query = f"MATCH (d:Disease {{name: '{top_disease}'}})-[:CONTRAINDICATED]->(f:Food) RETURN f.name AS food, f.reason AS reason"
    
    return PredictionResult(
        disease=top_disease,
        confidence=top_conf,
        explanation=explanation,
        dietary_precautions=dietary_precautions,
        cypher_query=cypher_query
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
