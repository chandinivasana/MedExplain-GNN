from __future__ import annotations
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
import json
import os

class MedicalGAT(torch.nn.Module):
    """GAT classifier for disease prediction from symptom graph features using HeteroConv."""

    def __init__(
        self,
        in_channels: int | None = None,
        hidden_channels: int = 256,
        out_channels: int | None = None,
        heads: int = 4,
        dropout: float = 0.35,
        num_node_features: int | None = None,
        num_classes: int | None = None,
    ):
        super().__init__()
        in_channels = in_channels if in_channels is not None else num_node_features
        
        if out_channels is None and num_classes is None:
            try:
                json_path = os.path.join(os.path.dirname(__file__), "index_to_disease.json")
                if not os.path.exists(json_path):
                    json_path = "index_to_disease.json"
                out_channels = len(json.load(open(json_path)))
            except Exception as e:
                out_channels = 6 # fallback
        else:
            out_channels = out_channels if out_channels is not None else num_classes
            
        if in_channels is None or out_channels is None:
            raise ValueError("MedicalGAT requires in_channels and out_channels.")

        self.dropout = dropout
        
        # Phase 3: Simplified HeteroConv for Demo
        # We focus on Symptom -> Disease path as it's the primary inference route
        self.gat1 = GATConv((in_channels, in_channels), hidden_channels, heads=heads, add_self_loops=False)
        self.gat2 = GATConv((hidden_channels * heads, hidden_channels * heads), out_channels, heads=1, add_self_loops=False)
        
        self.classifier = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        s_x = x_dict['Symptom']
        d_x = x_dict['Disease']
        edge_index = edge_index_dict[('Symptom', 'INDICATES', 'Disease')]

        s_x = F.dropout(s_x, p=self.dropout, training=self.training)
        d_x = F.dropout(d_x, p=self.dropout, training=self.training)
        
        # Layer 1
        h_d, (edge_idx, alpha) = self.gat1((s_x, d_x), edge_index, return_attention_weights=True)
        self.last_attention_weights = alpha.detach().cpu()
        self.last_edge_index = edge_idx.detach().cpu()
        
        h_d = F.elu(h_d)
        h_d = F.dropout(h_d, p=self.dropout, training=self.training)
        
        # Layer 2
        # Use h_d for both src and dst to keep shapes consistent for the second jump
        out_d = self.gat2((h_d, h_d), torch.stack([torch.arange(h_d.size(0)), torch.arange(h_d.size(0))]).to(h_d.device))
        out_d = F.elu(out_d)
        
        # Final Classification
        out_d = self.classifier(out_d)
            
        return {'Disease': out_d}
