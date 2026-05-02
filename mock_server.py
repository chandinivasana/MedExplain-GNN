import sys
import os
import torch
import pandas as pd
from unittest.mock import MagicMock, patch
import uvicorn

# Add ai_engine to path
sys.path.append(os.path.join(os.getcwd(), 'ai_engine'))

# --- MOCKING SERVICES ---

# Mock Neo4j
mock_driver = MagicMock()
mock_session = MagicMock()
mock_driver.session.return_value.__enter__.return_value = mock_session

# Mock Redis
mock_redis = MagicMock()

# Mock RQ
mock_queue = MagicMock()

# Mock MongoDB
mock_mongo = MagicMock()
mock_db = MagicMock()
mock_mongo.get_default_database.return_value = mock_db

# Load dataset for mock logic
df = pd.read_csv('data/cleaned_dataset.csv')
df['Disease'] = df['Disease'].str.strip()
unique_diseases = df['Disease'].unique().tolist()
edges = []
for _, row in df.iterrows():
    disease = row['Disease']
    for col in df.columns:
        if col.startswith('Symptom') and pd.notna(row[col]):
            symptom = str(row[col]).strip().lower().replace('_', ' ')
            edges.append({'symptom': symptom, 'disease': disease})

def mock_run_side_effect(query, **kwargs):
    if "MATCH (s:Symptom) RETURN" in query:
        return [{"name": s} for s in set(e['symptom'] for e in edges)]
    elif "MATCH (d:Disease) RETURN" in query:
        return [{"name": d} for d in unique_diseases]
    elif "MATCH (s:Symptom)-[r]->(d:Disease)" in query:
        return edges
    elif "MATCH (d:Disease {name: $name})-[:CONTRAINDICATED]" in query:
        return [{"food": "Aged Cheeses", "type": "Avoid"}, {"food": "Leafy Greens", "type": "Recommended"}]
    return []

mock_session.run.side_effect = mock_run_side_effect

# Patching everything before importing main
with patch('neo4j.GraphDatabase.driver', return_value=mock_driver), \
     patch('redis.Redis.from_url', return_value=mock_redis), \
     patch('rq.Queue', return_value=mock_queue), \
     patch('motor.motor_asyncio.AsyncIOMotorClient', return_value=mock_mongo):
    
    # Force mock environments
    os.environ["NEO4J_URI"] = "bolt://mock:7687"
    os.environ["REDIS_URL"] = "redis://mock:6379"
    os.environ["MONGO_URI"] = "mongodb://mock:27017"
    
    from main import app
    
    if __name__ == "__main__":
        print("\n" + "="*50)
        print("      AUTORUN: MEDEXPLAIN-GNN SIMULATED SERVER")
        print("="*50)
        print("Starting FastAPI with GAT Model & Patched Services...")
        uvicorn.run(app, host="127.0.0.1", port=8001)
