from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def populate(self):
        with self.driver.session() as session:
            session.execute_write(self._create_graph_data)

    @staticmethod
    def _create_graph_data(tx):
        # Create Symptoms
        tx.run("CREATE (s1:Symptom {name: 'Fever'})")
        tx.run("CREATE (s2:Symptom {name: 'Joint Pain'})")
        tx.run("CREATE (s3:Symptom {name: 'Headache'})")
        
        # Create Diseases
        tx.run("CREATE (d1:Disease {name: 'Influenza'})")
        tx.run("CREATE (d2:Disease {name: 'Dengue'})")
        
        # Create Food/Nutrients
        tx.run("CREATE (f1:Food {name: 'Sugary Foods', reasoning: 'Impairs immune response during viral infection'})")
        tx.run("CREATE (f2:Food {name: 'Salty Foods', reasoning: 'Promotes dehydration during high fever'})")

        # Create Relationships (Symptom -> Disease)
        tx.run("MATCH (s:Symptom {name: 'Fever'}), (d:Disease {name: 'Influenza'}) CREATE (s)-[:PRESENT_IN]->(d)")
        tx.run("MATCH (s:Symptom {name: 'Joint Pain'}), (d:Disease {name: 'Dengue'}) CREATE (s)-[:PRESENT_IN]->(d)")
        
        # Create Relationships (Disease -> Food Contraindications)
        tx.run("MATCH (d:Disease {name: 'Influenza'}), (f:Food {name: 'Sugary Foods'}) CREATE (d)-[:CONTRAINDICATED]->(f)")
        tx.run("MATCH (d:Disease {name: 'Dengue'}), (f:Food {name: 'Salty Foods'}) CREATE (d)-[:CONTRAINDICATED]->(f)")

if __name__ == "__main__":
    import os
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    kg = KnowledgeGraph(uri, user, password)
    kg.populate()
    kg.close()
    print(f"Neo4j database at {uri} populated with initial knowledge graph nodes and edges.")
