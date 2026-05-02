import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class MedicalGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=8): # Increased heads from 4 to 8
        super(MedicalGAT, self).__init__()
        torch.manual_seed(42)
        
        # Layer 1: Multi-head Graph Attention
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.3)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        
        # Layer 2: Refinement Graph Attention
        self.conv2 = GATConv(hidden_channels * heads, 32, heads=heads, dropout=0.3)
        self.bn2 = BatchNorm1d(32 * heads)
        
        # Output layer
        self.out_layer = Linear(32 * heads, num_classes)
        self.dropout = Dropout(p=0.6) # Increased dropout from 0.5 to 0.6

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
        x = self.dropout(x)
        
        # Final Classification
        x = self.out_layer(x)
        
        # THE FIX: Return raw logits for CrossEntropyLoss in train.py
        return x
