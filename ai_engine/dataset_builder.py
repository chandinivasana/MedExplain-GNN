import json
from pathlib import Path
import torch
import os
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase

def fetch_profiles_from_neo4j():
    """Fetch disease-symptom relationships from Neo4j."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    profiles = {}
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Query any relationship between Symptom and Disease
            result = session.run("""
                MATCH (s:Symptom)-[r]-(d:Disease)
                RETURN d.name AS disease, collect(DISTINCT s.name) AS symptoms
            """)
            for record in result:
                profiles[record["disease"]] = record["symptoms"]
        driver.close()
    except Exception as e:
        print(f"Warning: Could not fetch profiles from Neo4j: {e}. Using fallback.")
        profiles = {
            "Influenza": ["fever", "headache", "cough", "nausea"],
            "Dengue": ["fever", "joint pain", "headache", "nausea", "pain behind eyes"],
            "Asthma": ["cough", "shortness of breath", "wheezing", "chest tightness"],
            "Migraine": ["headache", "nausea", "sensitivity to light", "pain behind eyes"],
            "Food Poisoning": ["nausea", "vomiting", "abdominal pain", "diarrhea"],
            "Common Cold": ["cough", "sore throat", "runny nose", "headache"],
        }
    return profiles

def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values

class DatasetBuilder:
    def build_graph_data(self) -> HeteroData:
        return build_dataset(save=False)

def build_dataset(output_path: str = "graph_data.pt", save: bool = True) -> HeteroData:
    disease_profiles = fetch_profiles_from_neo4j()
    disease_names = sorted(list(disease_profiles.keys()))
    symptom_names = sorted({symptom for symptoms in disease_profiles.values() for symptom in symptoms})
    
    symptom_to_node = {symptom: index for index, symptom in enumerate(symptom_names)}
    disease_to_node = {disease: index for index, disease in enumerate(disease_names)}

    data = HeteroData()
    feature_size = len(symptom_names) + len(disease_names)

    # Symptom Nodes
    symptom_x = []
    for symptom in symptom_names:
        symptom_x.append(_one_hot(symptom_to_node[symptom], feature_size))
    data['Symptom'].x = torch.tensor(symptom_x, dtype=torch.float)

    # Disease Nodes
    disease_x = []
    disease_y = []
    train_mask = []
    val_mask = []

    for idx, disease in enumerate(disease_names):
        features = [0.0] * feature_size
        for symptom in disease_profiles[disease]:
            if symptom in symptom_to_node:
                features[symptom_to_node[symptom]] = 1.0
        # Add self-feature for disease
        features[len(symptom_names) + idx] = 1.0
        
        disease_x.append(features)
        disease_y.append(idx)
        train_mask.append(True)
        val_mask.append(False)

    # For a demo, we need at least some validation data to prevent NaNs
    # Let's use the last 10% for validation if possible
    if len(train_mask) > 10:
        for i in range(len(train_mask) - 5, len(train_mask)):
            train_mask[i] = False
            val_mask[i] = True

    data['Disease'].x = torch.tensor(disease_x, dtype=torch.float)
    data['Disease'].y = torch.tensor(disease_y, dtype=torch.long)
    data['Disease'].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data['Disease'].val_mask = torch.tensor(val_mask, dtype=torch.bool)

    data['Food'].x = torch.zeros((1, feature_size), dtype=torch.float)

    # Edges
    edge_pairs_sd = []
    for disease, symptoms in disease_profiles.items():
        d_idx = disease_to_node[disease]
        for symptom in symptoms:
            if symptom in symptom_to_node:
                s_idx = symptom_to_node[symptom]
                edge_pairs_sd.append((s_idx, d_idx))

    if edge_pairs_sd:
        data['Symptom', 'INDICATES', 'Disease'].edge_index = torch.tensor(edge_pairs_sd, dtype=torch.long).t().contiguous()
    else:
        data['Symptom', 'INDICATES', 'Disease'].edge_index = torch.empty((2, 0), dtype=torch.long)

    data['Disease', 'CONTRAINDICATED_WITH', 'Food'].edge_index = torch.empty((2, 0), dtype=torch.long)

    data.class_names = disease_names
    data.disease_names = disease_names # for backward compatibility
    data.symptom_names = symptom_names
    data.symptom_to_node = symptom_to_node
    data.num_classes = len(disease_names)
    data.num_symptoms = len(symptom_names)
    data.num_node_features_int = int(feature_size)

    if save:
        torch.save(data, output_path)
        Path("index_to_disease.json").write_text(
            json.dumps({index: disease for index, disease in enumerate(disease_names)}, indent=2)
        )
        Path("index_to_symptom.json").write_text(
            json.dumps({index: symptom for symptom, index in symptom_to_node.items()}, indent=2)
        )
    return data

if __name__ == "__main__":
    build_dataset()
