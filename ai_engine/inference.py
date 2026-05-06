from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from neo4j import GraphDatabase

try:
    from .model import MedicalGAT
except ImportError:
    from model import MedicalGAT


TEMPERATURE = 1.0
MAX_SYMPTOM_BOOST = 3.0
DEFAULT_CHECKPOINT_PATH = "best_model.pth"
DEFAULT_GRAPH_DATA_PATH = "graph_data.pt"


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_checkpoint_metadata(checkpoint_path: str, device: torch.device):
    return _safe_torch_load(Path(checkpoint_path), device)


def _load_model(checkpoint_path: str, device: torch.device) -> MedicalGAT:
    checkpoint = _load_checkpoint_metadata(checkpoint_path, device)
    model = MedicalGAT(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint.get("hidden_channels", 256),
        out_channels=checkpoint["out_channels"],
        heads=checkpoint.get("heads", 4),
        dropout=checkpoint.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_graph_data(graph_data_path: str, device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _safe_torch_load(Path(graph_data_path), device)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data.to(device)


def _metadata_value(*sources, keys):
    for source in sources:
        if source is None:
            continue
        for key in keys:
            if isinstance(source, dict) and key in source:
                return source[key]
            if hasattr(source, key):
                return getattr(source, key)
    return None


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _class_names(checkpoint, data, out_channels: int):
    names = _metadata_value(
        checkpoint,
        data,
        keys=("class_names", "disease_names", "idx_to_disease", "label_names"),
    )
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i), f"class_{i}"))) for i in range(out_channels)]
    if isinstance(names, (list, tuple)) and len(names) >= out_channels:
        return [str(name) for name in names[:out_channels]]
    return [f"class_{i}" for i in range(out_channels)]


def _symptom_to_node(checkpoint, data):
    mapping = _metadata_value(
        checkpoint,
        data,
        keys=("symptom_to_node", "symptom_node_map", "symptom_index", "node_name_to_idx"),
    )
    if isinstance(mapping, dict):
        return {_normalize_name(str(key)): int(value) for key, value in mapping.items()}

    node_names = _metadata_value(checkpoint, data, keys=("node_names", "symptom_names"))
    if isinstance(node_names, (list, tuple)):
        return {_normalize_name(str(name)): index for index, name in enumerate(node_names)}
    return {}


def symptom_indices_from_names(symptoms: Iterable[str], checkpoint, data) -> list[int]:
    mapping = _symptom_to_node(checkpoint, data)
    indices = []
    for symptom in symptoms:
        index = mapping.get(_normalize_name(symptom))
        if index is not None:
            indices.append(index)
    return sorted(set(indices))


def apply_symptom_signal(
    x: torch.Tensor,
    symptom_node_indices: Iterable[int],
    multiplier: float = MAX_SYMPTOM_BOOST,
) -> torch.Tensor:
    """Boost symptom nodes without creating extreme OOD feature magnitudes."""

    multiplier = min(float(multiplier), MAX_SYMPTOM_BOOST)
    boosted_x = x.clone()
    indices = list(symptom_node_indices)
    if not indices:
        return boosted_x

    node_idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    feature_mean = boosted_x.mean(dim=0, keepdim=True)
    feature_std = boosted_x.std(dim=0, keepdim=True).clamp_min(1e-6)

    boosted_x[node_idx] = boosted_x[node_idx] * multiplier
    lower = feature_mean - 4.0 * feature_std
    upper = feature_mean + 4.0 * feature_std
    return torch.max(torch.min(boosted_x, upper), lower)


@torch.no_grad()
def predict_disease_from_symptoms(
    symptoms: Iterable[str],
    graph_data_path: str = DEFAULT_GRAPH_DATA_PATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    temperature: float = TEMPERATURE,
    symptom_multiplier: float = MAX_SYMPTOM_BOOST,
    top_k: int = 3,
):
    if temperature < 0.5:
        raise ValueError("temperature below 0.5 makes predictions overconfident.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint_metadata(checkpoint_path, device)
    data = load_graph_data(graph_data_path, device)
    model = _load_model(checkpoint_path, device)

    symptom_node_indices = symptom_indices_from_names(symptoms, checkpoint, data)
    if not symptom_node_indices:
        raise ValueError("None of the extracted symptoms could be mapped to graph node indices.")

    x = apply_symptom_signal(data.x, symptom_node_indices, symptom_multiplier)
    logits = model(x, data.edge_index)

    node_idx = torch.as_tensor(symptom_node_indices, dtype=torch.long, device=device)
    pooled_logits = logits[node_idx].mean(dim=0)
    probabilities = F.softmax(pooled_logits / temperature, dim=0)

    names = _class_names(checkpoint, data, probabilities.numel())
    k = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=k)
    predictions = [
        (names[index.item()], float(value.item()))
        for value, index in zip(values, indices)
    ]

    best_disease, best_confidence = predictions[0]
    return best_disease, best_confidence, predictions


class GATInferenceEngine:
    def __init__(
        self,
        graph_data_path: str = DEFAULT_GRAPH_DATA_PATH,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        temperature: float = TEMPERATURE,
    ):
        self.graph_data_path = graph_data_path
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint = _load_checkpoint_metadata(checkpoint_path, self.device)
        self.data = load_graph_data(graph_data_path, self.device)
        self.model = _load_model(checkpoint_path, self.device)
        self.class_names = _class_names(self.checkpoint, self.data, self.checkpoint["out_channels"])
        self.symptom_mapping = _symptom_to_node(self.checkpoint, self.data)

        import os

        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def _dietary_precautions(self, disease: str) -> list[str]:
        precautions = []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Disease {name: $name})-[:CONTRAINDICATED]->(f:Food)
                    RETURN f.name AS food, 'Avoid' AS type
                    UNION
                    MATCH (d:Disease {name: $name})-[:RECOMMENDED]->(f:Food)
                    RETURN f.name AS food, 'Recommended' AS type
                    """,
                    name=disease,
                )
                for record in result:
                    precautions.append(f"{record['type']}: {record['food']}")
        except Exception:
            if disease == "Dengue":
                return ["Avoid: Salty Foods"]
            if disease == "Influenza":
                return ["Avoid: Sugary Foods"]
        return precautions

    def predict(self, symptoms: Iterable[str]):
        disease, confidence, predictions = predict_disease_from_symptoms(
            symptoms,
            graph_data_path=self.graph_data_path,
            checkpoint_path=self.checkpoint_path,
            temperature=self.temperature,
        )
        cypher_query = (
            f"MATCH (d:Disease {{name: '{disease}'}})"
            "-[:CONTRAINDICATED|RECOMMENDED]->(f:Food) RETURN f.name, labels(f)"
        )
        return predictions, self._dietary_precautions(disease), cypher_query


engine = None


def get_engine():
    global engine
    if engine is None:
        engine = GATInferenceEngine()
    return engine


@torch.no_grad()
def predict(
    data,
    symptom_node_indices: Iterable[int],
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    temperature: float = TEMPERATURE,
    symptom_multiplier: float = MAX_SYMPTOM_BOOST,
):
    if temperature < 0.5:
        raise ValueError("temperature below 0.5 makes predictions overconfident.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = _load_model(checkpoint_path, device)

    x = apply_symptom_signal(data.x, symptom_node_indices, symptom_multiplier)
    logits = model(x, data.edge_index)
    probabilities = F.softmax(logits / temperature, dim=1)
    confidence, prediction = probabilities.max(dim=1)

    return prediction.detach().cpu(), confidence.detach().cpu(), probabilities.detach().cpu()
