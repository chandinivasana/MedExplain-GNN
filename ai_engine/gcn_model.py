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
        self.conv2 = GCNConv(hidden_channels, 32)
        # The final layer MUST dynamically match the number of diseases in the DB
        self.out_layer = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output probability distribution across all diseases
        x = self.out_layer(x)
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
        print("Training GNN on graph data...")
        with self.driver.session() as session:
            symptoms_result = session.run("MATCH (s:Symptom) RETURN s.name AS name")
            diseases_result = session.run("MATCH (d:Disease) RETURN d.name AS name")
            edges_result = session.run("MATCH (s:Symptom)-[r:HAS_SYMPTOM]->(d:Disease) RETURN s.name AS symptom, d.name AS disease, r.weight AS weight")
            
            symptoms = [record["name"].lower() for record in symptoms_result]
            diseases = [record["name"] for record in diseases_result]
            edges_data = [(record["symptom"].lower(), record["disease"], record.get("weight", 0.8)) for record in edges_result]
        
        # Build index mappings
        for i, s in enumerate(symptoms):
            self.symptom_mapping[s] = i
        
        for i, d in enumerate(diseases):
            self.disease_mapping[d] = i
            self.reverse_disease_mapping[i] = d
            
        num_symptoms = len(self.symptom_mapping)
        num_diseases = len(self.disease_mapping)
        
        if num_symptoms == 0 or num_diseases == 0:
            print("Graph is empty. Please run seed_database.py first.")
            return

        # Prepare PyTorch Geometric Data representations
        total_nodes = num_symptoms + num_diseases
        x = torch.eye(total_nodes, dtype=torch.float)
        
        source_nodes = []
        target_nodes = []
        for s, d, w in edges_data:
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
        for epoch in range(100):
            optimizer.zero_grad()
            out = self.model(x, self.edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
        print(f"GNN successfully trained dynamically on {num_diseases} diseases and {num_symptoms} symptoms.")

    def predict(self, input_symptoms):
        """Performs GNN inference for a given array of symptoms and fetches dietary logic."""
        if not self.model:
            return [("Insufficient Data", 0.0)], [], "Graph empty"
            
        self.model.eval()
        num_symptoms = len(self.symptom_mapping)
        num_diseases = len(self.disease_mapping)
        total_nodes = num_symptoms + num_diseases
        x = torch.eye(total_nodes, dtype=torch.float)
        
        # Boost features of active input symptoms
        active_found = False
        for s in input_symptoms:
            s_lower = s.lower().replace("_", " ")
            if s_lower in self.symptom_mapping:
                idx = self.symptom_mapping[s_lower]
                x[idx] *= 5.0 
                active_found = True
        
        if not active_found:
             return [("No Matching Symptoms", 0.0)], [], "No symptoms matched graph nodes"
                
        with torch.no_grad():
            out = self.model(x, self.edge_index)
            
        # Extract probabilities for the disease nodes
        disease_out = out[num_symptoms:]
        probs = torch.exp(disease_out)
        
        # We take the diagonal if we want the node's own class confidence
        # But in multi-class node classification, we look at the softmax over classes
        # For inference, we can look at the average or sum of influences, 
        # or simply the node classification result for the disease nodes.
        disease_confidences = probs.diag()
        
        top_probs, top_indices = torch.topk(disease_confidences, k=min(3, num_diseases))
        predictions = []
        for p, idx in zip(top_probs, top_indices):
            predictions.append((self.reverse_disease_mapping[idx.item()], p.item()))
            
        top_disease = predictions[0][0]
        
        # Traverse graph for Contraindications/Recommendations
        dietary_precautions = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Disease {name: $name})-[:CONTRAINDICATED]->(f:Food) 
                RETURN f.name AS food, 'Avoid' AS type
                UNION
                MATCH (d:Disease {name: $name})-[:RECOMMENDED]->(f:Food) 
                RETURN f.name AS food, 'Recommended' AS type
            """, name=top_disease)
            for record in result:
                dietary_precautions.append(f"{record['type']}: {record['food']}")
        
        cypher_query = f"MATCH (d:Disease {{name: '{top_disease}'}})-[:CONTRAINDICATED|RECOMMENDED]->(f:Food) RETURN f.name, labels(f)"
                
        return predictions, dietary_precautions, cypher_query

# Singleton Engine
engine = None
def get_disease_prediction(symptoms):
    global engine
    if engine is None:
        engine = GNNInferenceEngine()
    
    predictions, dietary_precautions, cypher_query = engine.predict(symptoms)
    top_disease, top_conf = predictions[0]
    return top_disease, top_conf, predictions, dietary_precautions, cypher_query
