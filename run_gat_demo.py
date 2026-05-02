import sys
import os
import torch
import pandas as pd
from unittest.mock import MagicMock, patch

# Add ai_engine to path
sys.path.append(os.path.join(os.getcwd(), 'ai_engine'))

# --- MOCKING NEO4J ---
mock_driver = MagicMock()
mock_session = MagicMock()
mock_driver.session.return_value.__enter__.return_value = mock_session

# Load dataset for mock data
print("Loading cleaned dataset for mocking...")
df = pd.read_csv('data/cleaned_dataset.csv')
df['Disease'] = df['Disease'].str.strip()
unique_diseases = df['Disease'].unique().tolist()
unique_symptoms = set()
edges = []

for _, row in df.iterrows():
    disease = row['Disease']
    for col in df.columns:
        if col.startswith('Symptom') and pd.notna(row[col]):
            symptom = str(row[col]).strip().lower().replace('_', ' ')
            unique_symptoms.add(symptom)
            edges.append({'symptom': symptom, 'disease': disease})

symptoms_list = list(unique_symptoms)

def mock_run_side_effect(query, **kwargs):
    if "MATCH (s:Symptom) RETURN" in query:
        return [{"name": s} for s in symptoms_list]
    elif "MATCH (d:Disease) RETURN" in query:
        return [{"name": d} for d in unique_diseases]
    elif "MATCH (s:Symptom)-[r]->(d:Disease)" in query:
        return edges
    elif "MATCH (d:Disease {name: $name})-[:CONTRAINDICATED]" in query:
        return [{"food": "Aged Cheeses", "type": "Avoid"}, {"food": "Leafy Greens", "type": "Recommended"}]
    return []

mock_session.run.side_effect = mock_run_side_effect

# --- EXECUTION ---
with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
    with patch.dict(os.environ, {"NEO4J_URI": "bolt://mock:7687"}):
        print("\nStep 1: Building Dataset with BioBERT Embeddings...")
        from dataset_builder import DatasetBuilder
        builder = DatasetBuilder()
        data = builder.build_graph_data()
        torch.save(data, 'ai_engine/graph_data.pt')
        
        print("\nStep 2: Training GAT Model (Offline Training)...")
        from train import train
        train()
        
        print("\nStep 3: Running GAT Inference (Loaded Weights)...")
        from inference import GATInferenceEngine
        engine = GATInferenceEngine(temperature=1.2)
        
        test_text = ["high fever", "severe headache", "joint pain", "nausea"]
        print(f"\n[Test Input Symptoms]: {test_text}")
        
        predictions, dietary, cypher = engine.predict(test_text)
        
        print("\n" + "="*40)
        print("         GAT MODEL UPGRADE RESULTS")
        print("="*40)
        print(f"Predicted Disease: {predictions[0][0]}")
        print(f"Confidence Score:  {predictions[0][1]:.4f}")
        print("\nTop Predictions (Calibrated):")
        for d, c in predictions:
            print(f" - {d}: {c:.4f}")
        
        print("\nMLOps Persistence Check:")
        print(f" - graph_data.pt exists: {os.path.exists('ai_engine/graph_data.pt')}")
        print(f" - best_model.pth exists: {os.path.exists('ai_engine/best_model.pth')}")
        print("="*40)
