import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class MedicalGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=1):
        super(MedicalGAT, self).__init__()
        torch.manual_seed(42)
        
        # Simplify to a single powerful GAT layer to prevent over-smoothing
        self.conv1 = GATConv(num_node_features, 64, heads=1, dropout=0.1)
        self.out_layer = Linear(64, num_classes)

    def forward(self, x, edge_index):
        # We want to preserve the symptom features while allowing graph propagation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        return x # Return 64-d embeddings
