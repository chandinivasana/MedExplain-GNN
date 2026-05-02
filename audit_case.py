import torch
import torch.nn.functional as F
import os
import sys

# Add ai_engine to path
sys.path.append(os.path.join(os.getcwd(), 'ai_engine'))
from model import MedicalGAT

def audit_case():
    from unittest.mock import MagicMock, patch
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = []

    with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
        from inference import GATInferenceEngine
        engine = GATInferenceEngine()
        
        symptoms = ['increased thirst', 'blurred vision', 'unintended weight loss']
        print(f"Testing Symptoms: {symptoms}")
        
        predictions, dietary, cypher = engine.predict(symptoms)
        
        print("\n--- Similarity-Based Predictions ---")
        for d, c in predictions:
            print(f"  {d}: {c:.4f}")
        
        print(f"\nCypher Query: {cypher}")

if __name__ == "__main__":
    audit_case()
