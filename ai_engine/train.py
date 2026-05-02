import torch
import torch.nn.functional as F
from model import MedicalGAT
import os

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Load pre-built graph data
    if not os.path.exists('ai_engine/graph_data.pt'):
        print("graph_data.pt not found. Running dataset_builder.py...")
        from dataset_builder import DatasetBuilder
        builder = DatasetBuilder()
        data = builder.build_graph_data()
        torch.save(data, 'ai_engine/graph_data.pt')
    else:
        data = torch.load('ai_engine/graph_data.pt')

    data = data.to(device)
    
    # --- FIX 3: CLASS WEIGHTS (SQUARE ROOT INVERSE) ---
    disease_indices = range(data.num_symptoms, data.num_symptoms + data.num_classes)
    degrees = torch.zeros(data.num_classes, device=device)
    for i in range(data.edge_index.size(1)):
        tgt = data.edge_index[1, i].item()
        if tgt in disease_indices:
            degrees[tgt - data.num_symptoms] += 1
    
    # Penalize high-degree hubs moderately using square root inverse
    weights = 1.0 / (torch.sqrt(degrees) + 1e-6)
    weights = weights / weights.sum() * data.num_classes # Normalize
    
    model = MedicalGAT(num_node_features=data.x.size(1), hidden_channels=64, num_classes=data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        # Use the out_layer to get logits for training
        logits = model.out_layer(embeddings)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), 'ai_engine/best_model.pth')
    print("Model saved to ai_engine/best_model.pth")

if __name__ == "__main__":
    train()
