import csv
import os
from neo4j import GraphDatabase

# 1. Connect to your Neo4j instance
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

# Static mapping for demonstration purposes since CSV might not have diets
DIET_RULES = {
    "Hypertension": {"avoid": ["High Sodium Processed Meats"], "recommend": ["Leafy Greens"]},
    "Type 2 Diabetes": {"avoid": ["Refined Sugar"], "recommend": ["Whole Grains"]},
    "Migraine": {"avoid": ["Aged Cheeses"], "recommend": ["Water", "Ginger"]},
    "Gastroenteritis": {"avoid": ["Dairy Products"], "recommend": ["Bone Broth"]},
    "Hyperthyroidism": {"avoid": ["Excess Iodine"], "recommend": ["Cruciferous Vegetables"]},
}

def ingest_data(tx, disease_name, symptoms):
    # Create the Disease Node
    tx.run("MERGE (d:Disease {name: $disease_name})", disease_name=disease_name)
    
    # Create Symptom Nodes and link them to the Disease
    for symptom in symptoms:
        if not symptom or str(symptom).strip() == "":
            continue
            
        symptom_clean = str(symptom).strip().lower().replace("_", " ")
        
        query = """
        MERGE (s:Symptom {name: $symptom_name})
        WITH s
        MATCH (d:Disease {name: $disease_name})
        MERGE (d)<-[r:HAS_SYMPTOM {weight: 0.8}]-(s)
        """
        tx.run(query, symptom_name=symptom_clean, disease_name=disease_name)
    
    # Add Diet Rules
    if disease_name in DIET_RULES:
        rules = DIET_RULES[disease_name]
        for food in rules["avoid"]:
            tx.run("MERGE (f:Food {name: $food})", food=food)
            tx.run("""
                MATCH (d:Disease {name: $dis}), (f:Food {name: $food})
                MERGE (d)-[:CONTRAINDICATED]->(f)
            """, dis=disease_name, food=food)
        for food in rules["recommend"]:
            tx.run("MERGE (f:Food {name: $food})", food=food)
            tx.run("""
                MATCH (d:Disease {name: $dis}), (f:Food {name: $food})
                MERGE (d)-[:RECOMMENDED]->(f)
            """, dis=disease_name, food=food)

def build_graph():
    print("Loading real-world dataset...")
    data_path = "data/dataset.csv"
    if not os.path.exists(data_path):
        data_path = "/app/data/dataset.csv" # For Docker context
        
    with driver.session() as session:
        # Clear existing toy data
        print("Clearing old graph data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        print("Building new Knowledge Graph...")
        with open(data_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                disease = row['Disease'].strip()
                symptoms = [v for k, v in row.items() if k.startswith('Symptom') and v]
                session.execute_write(ingest_data, disease, symptoms)
                # print(f"Ingested: {disease}")
            
    print("Graph built successfully!")

if __name__ == "__main__":
    build_graph()
    driver.close()
