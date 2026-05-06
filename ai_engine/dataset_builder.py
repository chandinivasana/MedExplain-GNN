import json
from pathlib import Path

import torch
from torch_geometric.data import Data


DISEASE_PROFILES = {
    "Influenza": ["fever", "headache", "cough", "nausea"],
    "Dengue": ["fever", "joint pain", "headache", "nausea", "pain behind eyes"],
    "Asthma": ["cough", "shortness of breath", "wheezing", "chest tightness"],
    "Migraine": ["headache", "nausea", "sensitivity to light", "pain behind eyes"],
    "Food Poisoning": ["nausea", "vomiting", "abdominal pain", "diarrhea"],
    "Common Cold": ["cough", "sore throat", "runny nose", "headache"],
}


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


class DatasetBuilder:
    def build_graph_data(self) -> Data:
        return build_dataset(save=False)


def build_dataset(output_path: str = "graph_data.pt", save: bool = True) -> Data:
    disease_names = list(DISEASE_PROFILES)
    symptom_names = sorted({symptom for symptoms in DISEASE_PROFILES.values() for symptom in symptoms})
    symptom_to_node = {symptom: index for index, symptom in enumerate(symptom_names)}

    feature_size = len(symptom_names) + len(disease_names)
    x_rows = []
    y_rows = []
    train_mask = []
    val_mask = []
    test_mask = []
    edge_pairs = []

    for symptom in symptom_names:
        linked_diseases = [
            disease_index
            for disease_index, disease in enumerate(disease_names)
            if symptom in DISEASE_PROFILES[disease]
        ]
        primary_label = linked_diseases[0]
        if symptom in {"joint pain", "pain behind eyes"}:
            primary_label = disease_names.index("Dengue")

        x_rows.append(_one_hot(symptom_to_node[symptom], feature_size))
        y_rows.append(primary_label)
        train_mask.append(True)
        val_mask.append(False)
        test_mask.append(False)

    disease_node_offset = len(symptom_names)
    for disease_index, disease in enumerate(disease_names):
        profile = DISEASE_PROFILES[disease]
        features = [0.0] * feature_size
        for symptom in profile:
            features[symptom_to_node[symptom]] = 1.0
        features[len(symptom_names) + disease_index] = 1.0

        node_index = disease_node_offset + disease_index
        x_rows.append(features)
        y_rows.append(disease_index)
        train_mask.append(True)
        val_mask.append(False)
        test_mask.append(False)

        for symptom in profile:
            symptom_index = symptom_to_node[symptom]
            edge_pairs.append((symptom_index, node_index))
            edge_pairs.append((node_index, symptom_index))

    examples = [
        ("Dengue", ["fever", "joint pain", "pain behind eyes"]),
        ("Influenza", ["fever", "cough", "headache"]),
        ("Migraine", ["headache", "nausea", "pain behind eyes"]),
        ("Food Poisoning", ["nausea", "vomiting", "diarrhea"]),
        ("Asthma", ["cough", "wheezing", "shortness of breath"]),
        ("Common Cold", ["cough", "sore throat", "runny nose"]),
    ]

    for split_index, (disease, symptoms) in enumerate(examples * 3):
        disease_index = disease_names.index(disease)
        features = [0.0] * feature_size
        for symptom in symptoms:
            features[symptom_to_node[symptom]] = 1.0
        node_index = len(x_rows)
        x_rows.append(features)
        y_rows.append(disease_index)
        train_mask.append(split_index >= len(examples))
        val_mask.append(split_index < len(examples))
        test_mask.append(split_index < len(examples))

        for symptom in symptoms:
            symptom_index = symptom_to_node[symptom]
            edge_pairs.append((symptom_index, node_index))
            edge_pairs.append((node_index, symptom_index))

    for index in range(len(x_rows)):
        edge_pairs.append((index, index))

    data = Data(
        x=torch.tensor(x_rows, dtype=torch.float),
        edge_index=torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        y=torch.tensor(y_rows, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
    )
    data.class_names = disease_names
    data.disease_names = disease_names
    data.symptom_names = symptom_names
    data.symptom_to_node = symptom_to_node
    data.num_classes = len(disease_names)
    data.num_symptoms = len(symptom_names)

    if save:
        torch.save(data, output_path)
        Path("index_to_disease.json").write_text(
            json.dumps({index: disease for index, disease in enumerate(disease_names)}, indent=2)
        )
        print(f"Saved {output_path}")
        print("Saved index_to_disease.json")
        print(f"Nodes: {data.num_nodes}, edges: {data.num_edges}, classes: {len(disease_names)}")
    return data


if __name__ == "__main__":
    build_dataset()
