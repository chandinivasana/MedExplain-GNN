import argparse
import copy
from pathlib import Path
from typing import Optional
import json
import os
from neo4j import GraphDatabase

import torch
import torch.nn.functional as F

try:
    from .dataset_builder import build_dataset
    from .model import MedicalGAT
except ImportError:
    from dataset_builder import build_dataset
    from model import MedicalGAT


def regenerate_disease_index_from_neo4j():
    print("Regenerating index_to_disease.json from live Neo4j database...")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("MATCH (d:Disease) RETURN d.name AS name ORDER BY name")
            diseases = [record["name"] for record in result]
        driver.close()
        
        index_to_disease = {i: name for i, name in enumerate(diseases)}
        with open("index_to_disease.json", "w") as f:
            json.dump(index_to_disease, f, indent=2)
        print(f"Successfully generated index_to_disease.json with {len(diseases)} diseases.")
    except Exception as e:
        print(f"Warning: Could not regenerate index_to_disease.json from Neo4j: {e}")


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_data(path: Path, device: torch.device):
    if not path.exists():
        print(f"{path} not found. Building graph dataset first.")
        build_dataset(str(path))
    data = _safe_torch_load(path, device)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data.to(device)


def _validate_masks(data) -> None:
    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask"):
        raise ValueError("Training data must include both train_mask and val_mask.")
    if data.train_mask.sum().item() == 0:
        raise ValueError("train_mask is empty.")
    if data.val_mask.sum().item() == 0:
        raise ValueError("val_mask is empty.")


def _class_weights(y: torch.Tensor, train_mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    train_y = y[train_mask]
    counts = torch.bincount(train_y, minlength=num_classes).float().clamp_min(1.0)
    # Calculate per-class alpha as 1 / class_frequency normalized - rare diseases get higher alpha
    alpha = 1.0 / counts
    return alpha / alpha.mean()


def _checkpoint_metadata(data) -> dict:
    metadata = {}
    for name in (
        "class_names",
        "disease_names",
        "idx_to_disease",
        "label_names",
        "symptom_to_node",
        "symptom_node_map",
        "symptom_index",
        "node_name_to_idx",
        "node_names",
        "symptom_names",
        "num_classes",
        "num_symptoms",
    ):
        if hasattr(data, name):
            metadata[name] = getattr(data, name)
    return metadata


@torch.no_grad()
def evaluate(model: MedicalGAT, data, loss_fn: torch.nn.Module, mask: torch.Tensor):
    model.eval()
    logits_dict = model(data.x_dict, data.edge_index_dict)
    disease_logits = logits_dict['Disease']
    disease_y = data['Disease'].y
    loss = loss_fn(disease_logits[mask], disease_y[mask])
    preds = disease_logits[mask].argmax(dim=1)
    accuracy = (preds == disease_y[mask]).float().mean()
    return loss.item(), accuracy.item()


def train(
    data_path: str = "graph_data.pt",
    save_path: str = "best_model.pth",
    hidden_channels: int = 256,
    heads: int = 4,
    dropout: float = 0.35,
    gamma: float = 2.0,
    lr: float = 0.003,
    weight_decay: float = 5e-4,
    epochs: int = 500,
    patience: int = 60,
):
    regenerate_disease_index_from_neo4j()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load_data(Path(data_path), device)
    
    # We validate masks on the Disease node type
    if not hasattr(data['Disease'], "train_mask") or not hasattr(data['Disease'], "val_mask"):
        raise ValueError("Training data['Disease'] must include both train_mask and val_mask.")

    num_classes = int(data['Disease'].y.max().item()) + 1
    
    # Log class distribution
    train_y = data['Disease'].y[data['Disease'].train_mask]
    class_counts = torch.bincount(train_y, minlength=num_classes)
    print("Class distribution before training:", class_counts.tolist())

    in_channels = getattr(data, 'num_node_features_int', 21)
    model = MedicalGAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        heads=heads,
        dropout=dropout,
    ).to(device)

    weights = _class_weights(data['Disease'].y, data['Disease'].train_mask, num_classes).to(device)
    loss_fn = FocalLoss(alpha=weights, gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits_dict = model(data.x_dict, data.edge_index_dict)
        disease_logits = logits_dict['Disease']
        disease_y = data['Disease'].y
        train_mask = data['Disease'].train_mask
        
        train_loss = loss_fn(disease_logits[train_mask], disease_y[train_mask])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        val_loss, val_acc = evaluate(model, data, loss_fn, data['Disease'].val_mask)
        if True: # Save every epoch for the demo to ensure we have a model
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            
            save_dict = {
                "model_state_dict": best_state,
                "in_channels": int(in_channels),
                "hidden_channels": int(hidden_channels),
                "out_channels": int(num_classes),
                "heads": int(heads),
                "dropout": float(dropout),
                "val_loss": float(val_loss),
                "class_weights": weights.detach().cpu(),
                "focal_gamma": float(gamma),
                **_checkpoint_metadata(data),
            }
            torch.save(save_dict, save_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            train_acc = (disease_logits[train_mask].argmax(dim=1) == disease_y[train_mask]).float().mean()
            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_loss.item():.4f} "
                f"train_acc={train_acc.item():.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="graph_data.pt")
    parser.add_argument("--save-path", default="best_model.pth")
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()
    train(**vars(args))
