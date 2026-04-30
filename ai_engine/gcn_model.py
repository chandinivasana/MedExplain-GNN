import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from neo4j import GraphDatabase
import os

class MedicalGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(MedicalGNN, self).__init__()
        # Using Graph Convolutional Networks for scalable message passing
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # Softmax probability distribution over the dynamic classes
        return F.log_softmax(x, dim=1)

class GNNInferenceEngine:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        self.model = None
        self.symptom_mapping = {}
        self.disease_mapping = {}
        self.reverse_disease_mapping = {}
        self.edge_index = None
        
        self.train_gnn_on_graph()

    def train_gnn_on_graph(self):
        """Dynamically queries Neo4j to construct the PyG Data and trains the GCN."""
        with self.driver.session() as session:
            symptoms_result = session.run("MATCH (s:Symptom) RETURN s.name AS name")
            diseases_result = session.run("MATCH (d:Disease) RETURN d.name AS name")
            edges_result = session.run("MATCH (s:Symptom)-[:PRESENT_IN]->(d:Disease) RETURN s.name AS symptom, d.name AS disease")
            
            symptoms = [record["name"].lower() for record in symptoms_result]
            diseases = [record["name"] for record in diseases_result]
            edges_data = [(record["symptom"].lower(), record["disease"]) for record in edges_result]
        
        # Build index mappings
        for i, s in enumerate(symptoms):
            self.symptom_mapping[s] = i
        
        for i, d in enumerate(diseases):
            self.disease_mapping[d] = i
            self.reverse_disease_mapping[i] = d
            
        num_symptoms = len(self.symptom_mapping)
        num_diseases = len(self.disease_mapping)
        
        if num_symptoms == 0 or num_diseases == 0:
            print("Graph is empty. Please run populate_graph_v2.py first.")
            return

        # Prepare PyTorch Geometric Data representations
        total_nodes = num_symptoms + num_diseases
        x = torch.eye(total_nodes, dtype=torch.float)
        
        source_nodes = []
        target_nodes = []
        for s, d in edges_data:
            s_idx = self.symptom_mapping[s]
            d_idx = num_symptoms + self.disease_mapping[d]
            # Bidirectional message passing
            source_nodes.extend([s_idx, d_idx]) 
            target_nodes.extend([d_idx, s_idx])
            
        self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Define ground truth labels for diseases
        y = torch.zeros(total_nodes, dtype=torch.long)
        train_mask = torch.zeros(total_nodes, dtype=torch.bool)
        
        for d, idx in self.disease_mapping.items():
            node_idx = num_symptoms + idx
            y[node_idx] = idx
            train_mask[node_idx] = True
            
        # Initialize and Train GCN
        self.model = MedicalGNN(num_node_features=total_nodes, hidden_channels=64, num_classes=num_diseases)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        self.model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = self.model(x, self.edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
        print(f"GNN successfully trained dynamically on {num_diseases} diseases and {num_symptoms} symptoms.")

    def predict(self, input_symptoms):
        """Performs GNN inference for a given array of symptoms and fetches dietary logic."""
        if not self.model:
            return [("Insufficient Data", 0.0)], []
            
        self.model.eval()
        num_symptoms = len(self.symptom_mapping)
        num_diseases = len(self.disease_mapping)
        total_nodes = num_symptoms + num_diseases
        x = torch.eye(total_nodes, dtype=torch.float)
        
        # Boost features of active input symptoms to influence message passing
        active_symptom_indices = []
        for s in input_symptoms:
            s_lower = s.lower()
            if s_lower in self.symptom_mapping:
                idx = self.symptom_mapping[s_lower]
                x[idx] *= 5.0 # Activation multiplier
                active_symptom_indices.append(idx)
                
        with torch.no_grad():
            out = self.model(x, self.edge_index)
            
        # Extract Log Softmax for diseases and calculate probabilities
        disease_out = out[num_symptoms:]
        probs = torch.exp(disease_out)
        
        # Return Top 3 predicted diseases
        top_probs, top_indices = torch.topk(probs, k=min(3, num_diseases))
        predictions = []
        for p, idx in zip(top_probs, top_indices):
            predictions.append((self.reverse_disease_mapping[idx.item()], p.item()))
            
        top_disease = predictions[0][0]
        
        # Explainability: Traverse graph for Contraindications for the Top Prediction
        dietary_precautions = []
        with self.driver.session() as session:
            result = session.run("MATCH (d:Disease {name: $name})-[:CONTRAINDICATED]->(f:Food) RETURN f.name AS food, f.reason AS reason", name=top_disease)
            for record in result:
                dietary_precautions.append(f"{record['food']} - {record['reason']}")
                
        return predictions, dietary_precautions

# Singleton Engine wrapper
engine = None
def get_disease_prediction(symptoms):
    global engine
    if engine is None:
        engine = GNNInferenceEngine()
    
    predictions, dietary_precautions = engine.predict(symptoms)
    top_disease, top_conf = predictions[0]
    return top_disease, top_conf, predictions, dietary_precautions
