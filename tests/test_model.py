import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))
from model import MedicalGAT

def test_medical_gat_initialization():
    model = MedicalGAT(in_channels=768, hidden_channels=64, out_channels=50, heads=4)
    assert model is not None
    assert hasattr(model, 'gat1')
    assert hasattr(model, 'gat2')
    assert hasattr(model, 'classifier')

def test_medical_gat_forward_shape():
    num_symptoms = 20
    num_diseases = 10
    num_features = 768
    num_classes = 10
    
    model = MedicalGAT(in_channels=num_features, hidden_channels=64, out_channels=num_classes, heads=4)
    model.eval()

    x_dict = {
        'Symptom': torch.randn(num_symptoms, num_features),
        'Disease': torch.randn(num_diseases, num_features)
    }
    edge_index_dict = {
        ('Symptom', 'INDICATES', 'Disease'): torch.tensor([
            [0, 1, 2, 3, 4, 5, 0, 2],
            [1, 2, 3, 4, 5, 0, 3, 5]
        ], dtype=torch.long)
    }

    with torch.no_grad():
        out_dict = model(x_dict, edge_index_dict)

    assert 'Disease' in out_dict
    assert out_dict['Disease'].shape == (num_diseases, num_classes)
    assert not torch.isnan(out_dict['Disease']).any()

def test_medical_gat_attention_weights():
    num_symptoms = 5
    num_diseases = 3
    num_features = 768
    num_classes = 3
    
    model = MedicalGAT(in_channels=num_features, hidden_channels=32, out_channels=num_classes, heads=2)
    model.eval()

    x_dict = {
        'Symptom': torch.randn(num_symptoms, num_features),
        'Disease': torch.randn(num_diseases, num_features)
    }
    edge_index_dict = {
        ('Symptom', 'INDICATES', 'Disease'): torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    }

    with torch.no_grad():
        model(x_dict, edge_index_dict)

    assert hasattr(model, 'last_attention_weights')
    assert model.last_attention_weights is not None
