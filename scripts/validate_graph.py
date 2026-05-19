import os
import torch
from neo4j import GraphDatabase

def validate_graph():
    print("--- Phase 2: Dynamic Graph Validation ---")
    
    # 1. Check Neo4j Disease Count
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("MATCH (d:Disease) RETURN count(d) AS count")
            neo4j_count = result.single()["count"]
        driver.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return

    # 2. Check Model Output Channels
    checkpoint_path = "best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint {checkpoint_path} not found. Skipping validation.")
        return
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_count = checkpoint.get("out_channels", 0)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Neo4j Disease Count: {neo4j_count}")
    print(f"Model Output Channels: {model_count}")

    if neo4j_count != model_count:
        print("⚠️ WARNING: Graph-Model Mismatch detected!")
        print(f"The graph has {neo4j_count} diseases, but the model is trained for {model_count}.")
        print("Inference may crash or produce incorrect results. Please retrain the model.")
        # In a real CI/CD pipeline, you might exit 1 here
    else:
        print("✅ SUCCESS: Graph and Model are synchronized.")

if __name__ == "__main__":
    validate_graph()
