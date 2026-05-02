import torch
import torch.nn.functional as F
import os
from neo4j import GraphDatabase
from model import MedicalGAT
from dataset_builder import DatasetBuilder

class GATInferenceEngine:
    def __init__(self, temperature=1.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = temperature
        
        # Load Graph Data Metadata
        if not os.path.exists('ai_engine/graph_data.pt'):
            raise FileNotFoundError("graph_data.pt missing. Run train.py first.")
        
        self.data = torch.load('ai_engine/graph_data.pt', map_location=self.device)
        self.num_classes = self.data.num_classes
        self.num_symptoms = self.data.num_symptoms
        
        # Initialize Model
        self.model = MedicalGAT(num_node_features=self.data.x.size(1), hidden_channels=64, num_classes=self.num_classes).to(self.device)
        
        if os.path.exists('ai_engine/best_model.pth'):
            print("Loading weights from best_model.pth")
            self.model.load_state_dict(torch.load('ai_engine/best_model.pth', map_location=self.device))
        else:
            print("Warning: best_model.pth not found. Model is uninitialized.")
        
        self.model.eval()
        
        # Node mapping
        self.symptom_mapping = {name.lower(): i for i, name in enumerate(self.data.symptom_names)}
        self.reverse_disease_mapping = {i: name for i, name in enumerate(self.data.disease_names)}
        
        # Neo4j for dietary logic
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def predict(self, input_symptoms):
        # 1. Prepare Features
        x = self.data.x.clone()
        
        active_found = False
        for s in input_symptoms:
            s_lower = s.lower().replace("_", " ")
            if s_lower in self.symptom_mapping:
                idx = self.symptom_mapping[s_lower]
                x[idx] *= 2.0 
                active_found = True
        
        if not active_found:
            return [("No Matching Symptoms", 0.0)], [], "No symptoms matched graph nodes"

        # 2. Inference
        with torch.no_grad():
            logits = self.model(x, self.data.edge_index)
            # Apply Temperature Scaling to the disease nodes' logits
            disease_logits = logits[self.num_symptoms:]
            
            # THE FIX: We want a single distribution across all possible diseases.
            # We extract the score for the target class of each disease node (the diagonal)
            # and then apply Softmax across the entire vector of diseases.
            disease_scores = disease_logits.diag()
            probs = F.softmax(disease_scores / self.temperature, dim=0)
            
            top_probs, top_indices = torch.topk(probs, k=min(3, self.num_classes))
            
            predictions = []
            for p, idx in zip(top_probs, top_indices):
                predictions.append((self.reverse_disease_mapping[idx.item()], p.item()))

        top_disease = predictions[0][0]

        # 3. Dietary Logic from Neo4j
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

# Lifespan singleton
engine = None
def get_engine():
    global engine
    if engine is None:
        engine = GATInferenceEngine()
    return engine
