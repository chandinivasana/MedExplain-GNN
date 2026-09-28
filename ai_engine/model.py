from __future__ import annotations
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear

class MedicalGAT(torch.nn.Module):
    """GAT classifier for disease prediction from symptom graph features.

    Canonical architecture matching trained checkpoint best_model.pth:
      - conv1: GATConv(num_node_features, hidden_channels=64, heads=1, dropout=0.1)
      - out_layer: Linear(64, num_classes=50)
    """

    def __init__(
        self,
        num_node_features: int | None = None,
        hidden_channels: int = 64,
        num_classes: int | None = None,
        heads: int = 1,
        dropout: float = 0.1,
        in_channels: int | None = None,
        out_channels: int | None = None,
    ):
        super().__init__()
        num_node_features = num_node_features if num_node_features is not None else in_channels
        num_classes = num_classes if num_classes is not None else out_channels

        self.num_node_features = num_node_features if num_node_features is not None else 768
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes if num_classes is not None else 50
        self.dropout = dropout

        self.conv1 = GATConv(self.num_node_features, self.hidden_channels, heads=heads, dropout=self.dropout)
        self.out_layer = Linear(self.hidden_channels, self.num_classes)
        self.last_attention_weights = None
        self.last_edge_index = None

    def forward(
        self,
        x: torch.Tensor | dict,
        edge_index: torch.Tensor | dict,
        return_attention_weights: bool = False,
    ):
        if isinstance(x, dict):
            s_x = x.get('Symptom', torch.empty(0))
            d_x = x.get('Disease', torch.empty(0))
            x_tensor = torch.cat([s_x, d_x], dim=0) if s_x.numel() and d_x.numel() else (s_x if s_x.numel() else d_x)
            
            if isinstance(edge_index, dict):
                e = edge_index.get(('Symptom', 'INDICATES', 'Disease'))
                if e is not None:
                    edge_tensor = torch.stack([e[0], e[1] + s_x.size(0)])
                else:
                    edge_tensor = next(iter(edge_index.values()))
            else:
                edge_tensor = edge_index
        else:
            x_tensor = x
            edge_tensor = edge_index

        if return_attention_weights or not self.training:
            h, (edge_idx, alpha) = self.conv1(x_tensor, edge_tensor, return_attention_weights=True)
            self.last_attention_weights = alpha.detach().cpu()
            self.last_edge_index = edge_idx.detach().cpu()
        else:
            h = self.conv1(x_tensor, edge_tensor)

        h = F.elu(h)

        if isinstance(x, dict):
            num_sym = x['Symptom'].size(0) if 'Symptom' in x else 0
            d_emb = h[num_sym:]
            out_d = self.out_layer(d_emb)
            return {'Disease': out_d}

        if return_attention_weights:
            return h, (self.last_edge_index, self.last_attention_weights)

        return h
