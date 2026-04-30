import os
from neo4j import GraphDatabase

DISEASE_DATA = [
    {
        "disease": "Influenza",
        "symptoms": ["fever", "chills", "muscle aches", "cough", "congestion", "runny nose", "headache", "fatigue"],
        "foods_to_avoid": [{"name": "Sugary Foods", "reason": "Impairs immune response during viral infection"}],
        "foods_to_eat": [{"name": "Bone Broth", "reason": "Keeps you hydrated and provides nutrients"}]
    },
    {
        "disease": "Dengue",
        "symptoms": ["high fever", "severe headache", "pain behind the eyes", "joint pain", "muscle pain", "fatigue", "nausea", "vomiting", "skin rash"],
        "foods_to_avoid": [{"name": "Salty Foods", "reason": "Promotes dehydration during high fever"}],
        "foods_to_eat": [{"name": "Papaya Leaves", "reason": "Helps increase platelet count"}]
    },
    {
        "disease": "Malaria",
        "symptoms": ["fever", "chills", "profuse sweating", "headache", "nausea", "vomiting", "muscle pain", "fatigue"],
        "foods_to_avoid": [{"name": "Heavy Fried Foods", "reason": "Difficult to digest and strains the liver"}],
        "foods_to_eat": []
    },
    {
        "disease": "Type 2 Diabetes",
        "symptoms": ["increased thirst", "frequent urination", "increased hunger", "unintended weight loss", "fatigue", "blurred vision", "slow-healing sores", "frequent infections"],
        "foods_to_avoid": [{"name": "Refined Carbohydrates", "reason": "Causes rapid spikes in blood sugar"}],
        "foods_to_eat": []
    },
    {
        "disease": "Gastroenteritis",
        "symptoms": ["watery diarrhea", "abdominal cramps", "nausea", "vomiting", "muscle aches", "headache", "low-grade fever"],
        "foods_to_avoid": [{"name": "Dairy Products", "reason": "Can worsen diarrhea and stomach cramps"}],
        "foods_to_eat": []
    },
    {
        "disease": "Hypertension",
        "symptoms": ["headache", "shortness of breath", "nosebleeds", "flushing", "dizziness", "chest pain", "visual changes"],
        "foods_to_avoid": [{"name": "High Sodium Foods", "reason": "Increases blood pressure"}],
        "foods_to_eat": []
    },
    {
        "disease": "Migraine",
        "symptoms": ["throbbing pain", "pulsing pain", "sensitivity to light", "sensitivity to sound", "nausea", "vomiting"],
        "foods_to_avoid": [{"name": "Aged Cheeses", "reason": "Contains tyramine which triggers migraines"}],
        "foods_to_eat": []
    },
    {
        "disease": "COVID-19",
        "symptoms": ["fever", "cough", "tiredness", "loss of taste", "loss of smell", "shortness of breath", "muscle aches"],
        "foods_to_avoid": [{"name": "Processed Meats", "reason": "Increases inflammation"}],
        "foods_to_eat": []
    },
    {
        "disease": "Asthma",
        "symptoms": ["shortness of breath", "chest tightness", "wheezing", "coughing attacks"],
        "foods_to_avoid": [{"name": "Sulfites", "reason": "Can trigger asthma symptoms"}],
        "foods_to_eat": []
    },
    {
        "disease": "Anemia",
        "symptoms": ["fatigue", "weakness", "pale skin", "chest pain", "cold hands and feet", "brittle nails"],
        "foods_to_avoid": [{"name": "Tea and Coffee", "reason": "Tannins interfere with iron absorption"}],
        "foods_to_eat": []
    },
    {
        "disease": "Celiac Disease",
        "symptoms": ["diarrhea", "fatigue", "weight loss", "bloating", "gas", "abdominal pain", "nausea"],
        "foods_to_avoid": [{"name": "Gluten", "reason": "Triggers an autoimmune response damaging the small intestine"}],
        "foods_to_eat": []
    },
    {
        "disease": "Peptic Ulcer",
        "symptoms": ["burning stomach pain", "feeling of fullness", "bloating", "belching", "fatigue", "heartburn", "nausea"],
        "foods_to_avoid": [{"name": "Spicy Foods", "reason": "Irritates the stomach lining and ulcer"}],
        "foods_to_eat": []
    },
    {
        "disease": "Rheumatoid Arthritis",
        "symptoms": ["tender joints", "warm joints", "swollen joints", "joint stiffness", "fatigue", "fever", "loss of appetite"],
        "foods_to_avoid": [{"name": "Red Meat", "reason": "Contains saturated fats that increase inflammation"}],
        "foods_to_eat": []
    },
    {
        "disease": "Hypothyroidism",
        "symptoms": ["fatigue", "increased sensitivity to cold", "constipation", "dry skin", "weight gain", "puffy face", "muscle weakness"],
        "foods_to_avoid": [{"name": "Soy Products", "reason": "Can interfere with thyroid hormone absorption"}],
        "foods_to_eat": []
    },
    {
        "disease": "Hyperthyroidism",
        "symptoms": ["unintentional weight loss", "rapid heartbeat", "irregular heartbeat", "palpitations", "increased appetite", "nervousness", "anxiety", "irritability", "tremor", "sweating"],
        "foods_to_avoid": [{"name": "Excess Iodine", "reason": "Can worsen hyperthyroidism by overstimulating the thyroid"}],
        "foods_to_eat": []
    },
    {
        "disease": "Gout",
        "symptoms": ["intense joint pain", "lingering discomfort", "inflammation", "redness", "limited range of motion"],
        "foods_to_avoid": [{"name": "Purine-rich Foods", "reason": "Increases uric acid levels in the blood"}],
        "foods_to_eat": []
    },
    {
        "disease": "Chronic Kidney Disease",
        "symptoms": ["nausea", "vomiting", "loss of appetite", "fatigue", "weakness", "sleep problems", "changes in urine output", "decreased mental sharpness", "muscle twitches"],
        "foods_to_avoid": [{"name": "High Potassium Foods", "reason": "Kidneys cannot effectively filter excess potassium"}],
        "foods_to_eat": []
    },
    {
        "disease": "Tuberculosis",
        "symptoms": ["persistent cough", "chest pain", "coughing up blood", "fatigue", "weight loss", "fever", "night sweats", "chills"],
        "foods_to_avoid": [{"name": "Alcohol", "reason": "Can interact with TB medications and damage the liver"}],
        "foods_to_eat": []
    },
    {
        "disease": "Pneumonia",
        "symptoms": ["chest pain", "confusion", "cough with phlegm", "fatigue", "fever", "sweating", "chills", "nausea", "vomiting", "shortness of breath"],
        "foods_to_avoid": [{"name": "Dairy Products", "reason": "May thicken mucus and worsen congestion"}],
        "foods_to_eat": []
    },
    {
        "disease": "Lupus",
        "symptoms": ["fatigue", "fever", "joint pain", "stiffness", "swelling", "butterfly rash", "skin lesions", "shortness of breath", "chest pain"],
        "foods_to_avoid": [{"name": "Alfalfa Sprouts", "reason": "Contains an amino acid that can trigger lupus flares"}],
        "foods_to_eat": []
    }
]

class KnowledgeGraphV2:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def populate(self):
        with self.driver.session() as session:
            # Clear existing graph completely to prevent duplication
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
            session.execute_write(self._create_graph_data)

    @staticmethod
    def _create_graph_data(tx):
        for data in DISEASE_DATA:
            disease_name = data["disease"]
            tx.run("MERGE (d:Disease {name: $name})", name=disease_name)
            
            for sym in data["symptoms"]:
                tx.run("MERGE (s:Symptom {name: $name})", name=sym.lower())
                tx.run(
                    "MATCH (s:Symptom {name: $sym}), (d:Disease {name: $dis}) "
                    "MERGE (s)-[:PRESENT_IN]->(d)",
                    sym=sym.lower(), dis=disease_name
                )
                
            for food in data["foods_to_avoid"]:
                tx.run(
                    "MERGE (f:Food {name: $name, reason: $reason})", 
                    name=food["name"], reason=food["reason"]
                )
                tx.run(
                    "MATCH (d:Disease {name: $dis}), (f:Food {name: $food}) "
                    "MERGE (d)-[:CONTRAINDICATED]->(f)",
                    dis=disease_name, food=food["name"]
                )
                
            for food in data["foods_to_eat"]:
                tx.run(
                    "MERGE (f:Food {name: $name, reason: $reason})", 
                    name=food["name"], reason=food["reason"]
                )
                tx.run(
                    "MATCH (d:Disease {name: $dis}), (f:Food {name: $food}) "
                    "MERGE (d)-[:RECOMMENDED]->(f)",
                    dis=disease_name, food=food["name"]
                )

if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    kg = KnowledgeGraphV2(uri, user, password)
    kg.populate()
    kg.close()
    print(f"Neo4j database at {uri} populated with 20 diseases and relationships.")
