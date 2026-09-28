import pytest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_engine'))
from model import MedicalGAT

def test_medical_gat_initialization():
    model = MedicalGAT(num_node_features=768, hidden_channels=64, num_classes=50, heads=1)
    assert model is not None
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'out_layer')
    assert model.conv1.in_channels == 768
    assert model.conv1.out_channels == 64
    assert model.out_layer.out_features == 50

def test_medical_gat_forward_shape():
    num_nodes = 30
    num_features = 768
    num_classes = 10
    
    model = MedicalGAT(num_node_features=num_features, hidden_channels=64, num_classes=num_classes, heads=1)
    model.eval()

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 0, 2],
        [1, 2, 3, 4, 5, 0, 3, 5]
    ], dtype=torch.long)

    with torch.no_grad():
        out = model(x, edge_index)

    assert out.shape == (num_nodes, 64)
    assert not torch.isnan(out).any()

    logits = model.out_layer(out)
    assert logits.shape == (num_nodes, num_classes)

def test_medical_gat_attention_weights():
    num_nodes = 8
    num_features = 768
    num_classes = 3
    
    model = MedicalGAT(num_node_features=num_features, hidden_channels=32, num_classes=num_classes, heads=1)
    model.eval()

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    with torch.no_grad():
        out, (edge_idx, alpha) = model(x, edge_index, return_attention_weights=True)

    assert hasattr(model, 'last_attention_weights')
    assert model.last_attention_weights is not None
    assert alpha.shape[1] == 1  # 1 head
    assert alpha.shape[0] == edge_idx.shape[1]

def test_medical_gat_checkpoint_loading():
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ai_engine', 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model = MedicalGAT(num_node_features=768, hidden_channels=64, num_classes=50, heads=1)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        assert set(model.state_dict().keys()) == set(state_dict.keys())
