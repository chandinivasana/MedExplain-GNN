import sys
import os
import pandas as pd
from unittest.mock import MagicMock, patch

# Add ai_engine to path so we can import modules
sys.path.append(os.path.join(os.getcwd(), 'ai_engine'))

# Mock Neo4j before importing gcn_model because it initializes the driver on import/singleton
mock_driver = MagicMock()
mock_session = MagicMock()
mock_driver.session.return_value.__enter__.return_value = mock_session

print("Loading dataset for mocking...")
df = pd.read_csv('data/dataset.csv')
# Clean up disease names (remove trailing spaces if any)
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
            edges.append({'symptom': symptom, 'disease': disease, 'weight': 0.8})

symptoms_list = list(unique_symptoms)
print(f"Loaded {len(unique_diseases)} diseases and {len(symptoms_list)} unique symptoms.")

def mock_run_side_effect(query, **kwargs):
    if "MATCH (s:Symptom) RETURN" in query:
        return [{"name": s} for s in symptoms_list]
    elif "MATCH (d:Disease) RETURN" in query:
        return [{"name": d} for d in unique_diseases]
    elif "MATCH (s:Symptom)-[r:HAS_SYMPTOM]->(d:Disease)" in query:
        return edges
    elif "MATCH (d:Disease {name: $name})-[:CONTRAINDICATED]" in query:
        # Return dummy precautions for the predicted disease
        return [{"food": "Aged Cheeses", "type": "Avoid"}, {"food": "Leafy Greens", "type": "Recommended"}]
    return []

mock_session.run.side_effect = mock_run_side_effect

# Patching the driver before importing gcn_model
with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
    # Also need to mock environment variables that gcn_model uses
    with patch.dict(os.environ, {"NEO4J_URI": "bolt://mock:7687"}):
        import gcn_model
        from symptom_extractor import SymptomExtractor
        
        print("\nInitializing Symptom Extractor (BioBERT)...")
        # SymptomExtractor might download a model, which is fine as long as we have internet
        extractor = SymptomExtractor()
        
        test_text = "I have a high fever, severe headache, and joint pain. I feel very tired and have nausea."
        print(f"\n[Test Input]: {test_text}")
        
        extracted_symptoms = extractor.extract(test_text)
        print(f"[Extracted Symptoms]: {extracted_symptoms}")
        
        if extracted_symptoms:
            print("\nRunning GNN Inference (Training on mock graph first)...")
            # get_disease_prediction will initialize the engine and train it on our mock data
            top_disease, top_conf, all_predictions, dietary_precautions, cypher_query = gcn_model.get_disease_prediction(extracted_symptoms)
            
            print("\n" + "="*40)
            print("         MODEL INFERENCE RESULTS")
            print("="*40)
            print(f"Predicted Disease: {top_disease}")
            print(f"Confidence Score:  {top_conf:.4f}")
            print("\nTop 3 Predictions:")
            for d, c in all_predictions:
                print(f" - {d}: {c:.4f}")
            
            print("\nDietary Precautions:")
            for p in dietary_precautions:
                print(f" - {p}")
            
            print("\nGenerated Cypher Query for Explainability:")
            print(f"  {cypher_query}")
            print("="*40)
        else:
            print("No symptoms extracted. Try a different input.")
