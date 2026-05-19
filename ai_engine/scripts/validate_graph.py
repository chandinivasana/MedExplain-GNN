import json
import sys
import os
from neo4j import GraphDatabase

def validate():
    print("Validating Neo4j graph alignment with model configuration...")
    
    # Read model configuration (index_to_disease.json)
    try:
        with open("index_to_disease.json", "r") as f:
            index_mapping = json.load(f)
        json_count = len(index_mapping)
    except Exception as e:
        print(f"Error reading index_to_disease.json: {e}")
        json_count = 0

    # Read Neo4j total classes
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("MATCH (d:Disease) RETURN count(d) AS Total_Classes")
            total_classes = result.single()["Total_Classes"]
        driver.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

    print(f"Neo4j Total_Classes: {total_classes}")
    print(f"JSON index count: {json_count}")

    if total_classes != json_count:
        print("Validation failed: Neo4j disease count differs from index_to_disease.json!")
        print("RETRAINING REQUIRED. Please regenerate the mapping and retrain the model.")
        sys.exit(1)
        
    print("Graph validation passed.")

if __name__ == "__main__":
    validate()
