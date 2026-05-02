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
    model = MedicalGAT(num_node_features=data.x.size(1), hidden_channels=64, num_classes=data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=15)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save model
    torch.save(model.state_dict(), 'ai_engine/best_model.pth')
    print("Model saved to ai_engine/best_model.pth")

if __name__ == "__main__":
    train()
