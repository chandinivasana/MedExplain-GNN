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

# Load cleaned dataset
print("Loading sanitized dataset for rebuild...")
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
    return []

mock_session.run.side_effect = mock_run_side_effect

# --- REBUILD AND TRAIN ---
with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
    with patch.dict(os.environ, {"NEO4J_URI": "bolt://mock:7687"}):
        from dataset_builder import DatasetBuilder
        from train import train
        
        print("\n[Step 1] Rebuilding graph_data.pt with sanitized nodes...")
        builder = DatasetBuilder()
        data = builder.build_graph_data()
        torch.save(data, 'ai_engine/graph_data.pt')
        
        print("\n[Step 2] Retraining GAT model with sanitized logic...")
        train()
        
        print("\nRebuild and retrain complete.")
