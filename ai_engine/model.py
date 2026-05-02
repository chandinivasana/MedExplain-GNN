import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class MedicalGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=4):
        super(MedicalGAT, self).__init__()
        torch.manual_seed(42)
        
        # Layer 1: Multi-head Graph Attention
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.2)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        
        # Layer 2: Refinement Graph Attention
        self.conv2 = GATConv(hidden_channels * heads, 32, heads=1, dropout=0.2)
        self.bn2 = BatchNorm1d(32)
        
        # Output layer
        self.out_layer = Linear(32, num_classes)
        self.dropout = Dropout(p=0.5)

    def forward(self, x, edge_index):
        # First GAT Layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT Layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        # Final Classification
        x = self.out_layer(x)
        
        # Return log_softmax for NLLLoss or raw logits for CrossEntropy
        return F.log_softmax(x, dim=1)
