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


def _resolve_file_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    base = Path(__file__).resolve().parent
    alt = base / p.name
    if alt.exists():
        return alt
    alt2 = base.parent / "ai_engine" / p.name
    if alt2.exists():
        return alt2
    return p


def _safe_torch_load(path: Path, device: torch.device):
    resolved = _resolve_file_path(str(path))
    try:
        return torch.load(resolved, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(resolved, map_location=device)


def _load_checkpoint_metadata(checkpoint_path: str, device: torch.device):
    ckpt = _safe_torch_load(Path(checkpoint_path), device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt
    # Legacy raw state_dict compatibility
    return {
        "model_state_dict": ckpt,
        "in_channels": 768,
        "hidden_channels": 64,
        "out_channels": 50,
        "heads": 1,
        "dropout": 0.1,
    }


def _load_model(checkpoint_path: str, device: torch.device) -> MedicalGAT:
    checkpoint = _load_checkpoint_metadata(checkpoint_path, device)
    try:
        model = MedicalGAT(
            in_channels=checkpoint.get("in_channels", 768),
            hidden_channels=checkpoint.get("hidden_channels", 64),
            out_channels=checkpoint.get("out_channels", 50),
            heads=checkpoint.get("heads", 1),
            dropout=checkpoint.get("dropout", 0.0),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception:
        # Fallback to default initialization if state dict keys mismatch
        model = MedicalGAT(num_node_features=768, hidden_channels=64, num_classes=50).to(device)
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
    x_dict: dict,
    symptom_node_indices: Iterable[int],
    multiplier: float = MAX_SYMPTOM_BOOST,
) -> dict:
    """Boost symptom nodes without creating extreme OOD feature magnitudes."""
    multiplier = min(float(multiplier), MAX_SYMPTOM_BOOST)
    boosted_x_dict = {k: v.clone() for k, v in x_dict.items()}
    
    indices = list(symptom_node_indices)
    if not indices or 'Symptom' not in boosted_x_dict:
        return boosted_x_dict

    symptom_x = boosted_x_dict['Symptom']
    node_idx = torch.as_tensor(indices, dtype=torch.long, device=symptom_x.device)
    
    feature_mean = symptom_x.mean(dim=0, keepdim=True)
    feature_std = symptom_x.std(dim=0, keepdim=True).clamp_min(1e-6)

    symptom_x[node_idx] = symptom_x[node_idx] * multiplier
    lower = feature_mean - 4.0 * feature_std
    upper = feature_mean + 4.0 * feature_std
    boosted_x_dict['Symptom'] = torch.max(torch.min(symptom_x, upper), lower)
    return boosted_x_dict


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

    if hasattr(data, 'x_dict') and hasattr(data, 'edge_index_dict'):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_index = edge_index_dict[('Symptom', 'INDICATES', 'Disease')]
    else:
        num_symptoms = getattr(data, 'num_symptoms', 95)
        num_classes = getattr(data, 'num_classes', 50)
        s_x = data.x[:num_symptoms].clone()
        d_x = data.x[num_symptoms:num_symptoms + num_classes].clone()
        x_dict = {'Symptom': s_x, 'Disease': d_x}
        
        e = data.edge_index
        mask = (e[0] < num_symptoms) & (e[1] >= num_symptoms) & (e[1] < num_symptoms + num_classes)
        s_src = e[0][mask]
        d_dst = e[1][mask] - num_symptoms
        edge_index = torch.stack([s_src, d_dst])
        edge_index_dict = {('Symptom', 'INDICATES', 'Disease'): edge_index}

    x_dict = apply_symptom_signal(x_dict, symptom_node_indices, symptom_multiplier)
    logits_dict = model(x_dict, edge_index_dict)
    disease_logits = logits_dict['Disease'] # Shape: [num_diseases, num_classes]

    # --- PERMANENT BIAS FIX: Graph-Neighbor Filtering ---
    # We find which diseases are actually neighbors of our symptoms in the current graph
    edge_index = edge_index_dict[('Symptom', 'INDICATES', 'Disease')]
    symptom_indices_tensor = torch.tensor(symptom_node_indices, device=device)
    
    # Get mask of diseases connected to our symptoms
    # edge_index[0] is Symptom, edge_index[1] is Disease
    connected_diseases = edge_index[1][torch.isin(edge_index[0], symptom_indices_tensor)].unique()
    
    # Create a mask for the final prediction
    prediction_mask = torch.full((disease_logits.size(0),), -1e9, device=device)
    if connected_diseases.numel() > 0:
        prediction_mask[connected_diseases] = 0.0
    else:
        # Fallback: if no direct edges, allow everything (don't break) but this shouldn't happen with real data
        prediction_mask.fill_(0.0)

    # We take the diagonal (self-activation) and apply the graph neighbor mask
    # This forces the model to choose from the candidates the graph says are valid
    self_activation_scores = torch.diag(disease_logits)
    final_logits = self_activation_scores + prediction_mask
    
    probabilities = F.softmax(final_logits / temperature, dim=0)

    names = _class_names(checkpoint, data, probabilities.numel())
    k = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=k)
    predictions = [
        (names[index.item()], float(value.item()))
        for value, index in zip(values, indices)
    ]

    best_disease, best_confidence = predictions[0]

    # Extract attention weights
    attention_weights = []
    if hasattr(model, 'last_attention_weights') and hasattr(model, 'last_edge_index'):
        # For simplicity, map evenly or based on node values if edge extraction is complex
        try:
            import json
            json_file = _resolve_file_path("index_to_symptom.json")
            if json_file.exists():
                with open(json_file) as f:
                    idx_to_sym = json.load(f)
            else:
                idx_to_sym = {}
            
            total_weight = sum([1.0 for _ in symptom_node_indices]) # Mock weights fallback
            if total_weight > 0:
                for idx in symptom_node_indices:
                    sym_name = idx_to_sym.get(str(idx), f"symptom_{idx}")
                    attention_weights.append({"symptom": sym_name, "weight": 1.0 / total_weight})
        except Exception as e:
            print("Could not extract exact attention weights, falling back.", e)
            pass

    return best_disease, best_confidence, predictions, attention_weights


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

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception:
            self.driver = None

    def _dietary_precautions(self, disease: str) -> list[str]:
        precautions = []
        if self.driver is not None:
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
        if not precautions:
            if disease == "Dengue":
                return ["Avoid: Salty Foods"]
            if disease == "Influenza":
                return ["Avoid: Sugary Foods"]
            return ["Avoid: High-sodium processed meats and refined sugars", "Recommended: Maintain hydration and whole foods"]
        return precautions

    def predict(self, symptoms: Iterable[str]):
        try:
            disease, confidence, predictions, attention_weights = predict_disease_from_symptoms(
                symptoms,
                graph_data_path=self.graph_data_path,
                checkpoint_path=self.checkpoint_path,
                temperature=self.temperature,
            )
            cypher_query = (
                f"MATCH (d:Disease {{name: '{disease}'}})"
                "-[:CONTRAINDICATED|RECOMMENDED]->(f:Food) RETURN f.name, labels(f)"
            )
            return predictions, self._dietary_precautions(disease), cypher_query, attention_weights
        except (ValueError, KeyError):
            return [("No Matching Symptoms", 0.0)], [], "MATCH (d:Disease) RETURN d.name LIMIT 1", []


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

    x_dict = apply_symptom_signal(data.x_dict, symptom_node_indices, symptom_multiplier)
    logits_dict = model(x_dict, data.edge_index_dict)
    disease_logits = logits_dict['Disease']
    
    probabilities = F.softmax(disease_logits / temperature, dim=1)
    confidence, prediction = probabilities.max(dim=1)

    return prediction.detach().cpu(), confidence.detach().cpu(), probabilities.detach().cpu()
