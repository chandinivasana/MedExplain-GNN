import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))
from symptom_extractor import SymptomExtractor

@pytest.fixture(scope="module")
def extractor():
    return SymptomExtractor()

def test_symptom_extractor_init(extractor):
    assert extractor is not None
    assert len(extractor.common_symptoms) > 15

def test_extract_single_symptom(extractor):
    text = "The patient reports having a high fever for three days."
    symptoms = extractor.extract(text)
    assert any("fever" in s.lower() for s in symptoms)

def test_extract_multiple_symptoms(extractor):
    text = "Patient experiences severe headache, joint pain, and nausea."
    symptoms = extractor.extract(text)
    assert len(symptoms) >= 2
    symptoms_joined = " ".join(symptoms).lower()
    assert "headache" in symptoms_joined
    assert "joint pain" in symptoms_joined

def test_extract_empty_string(extractor):
    symptoms = extractor.extract("")
    assert symptoms == []

def test_extract_case_insensitivity(extractor):
    text = "PATIENT SUFFERING FROM COUGH AND FATIGUE"
    symptoms = extractor.extract(text)
    symptoms_joined = " ".join(symptoms).lower()
    assert "cough" in symptoms_joined or "fatigue" in symptoms_joined
