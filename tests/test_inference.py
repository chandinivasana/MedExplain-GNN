import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))
from inference import GATInferenceEngine

@pytest.fixture(scope="module")
def engine():
    return GATInferenceEngine()

def test_inference_engine_init(engine):
    assert engine is not None
    assert len(engine.class_names) > 0
    assert len(engine.symptom_mapping) > 0

def test_predict_with_valid_symptoms(engine):
    symptoms = ["fever", "cough"]
    predictions, dietary_precautions, cypher_query, attention_weights = engine.predict(symptoms)

    assert len(predictions) > 0
    top_disease, top_prob = predictions[0]
    assert isinstance(top_disease, str)
    assert 0.0 <= top_prob <= 1.0

    # Probabilities should be strictly sorted in descending order
    probs = [p[1] for p in predictions]
    assert probs == sorted(probs, reverse=True)

    # Dietary precautions must be returned (either via Neo4j or clinical fallback)
    assert len(dietary_precautions) > 0
    assert any("Avoid" in item or "Recommended" in item for item in dietary_precautions)

    # Cypher query validation
    assert cypher_query.startswith("MATCH (d:Disease")
    assert top_disease in cypher_query

def test_predict_with_no_matching_symptoms(engine):
    symptoms = ["completely_fake_symptom_xyz_123"]
    predictions, dietary_precautions, cypher_query, attention_weights = engine.predict(symptoms)

    assert predictions[0][0] == "No Matching Symptoms"
    assert predictions[0][1] == 0.0
    assert dietary_precautions == []
