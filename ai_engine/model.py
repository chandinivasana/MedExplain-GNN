import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MedicalGAT(torch.nn.Module):
    """GAT classifier for disease prediction from symptom graph features."""

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
        out_channels = out_channels if out_channels is not None else num_classes
        if in_channels is None or out_channels is None:
            raise ValueError("MedicalGAT requires in_channels and out_channels.")
        if hidden_channels < 256:
            raise ValueError("MedicalGAT should use hidden_channels >= 256 for 50+ disease classes.")

        self.dropout = dropout
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        return self.classifier(x)
