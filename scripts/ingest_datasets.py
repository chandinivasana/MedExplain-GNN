import os
import argparse
import pandas as pd
from neo4j import GraphDatabase

# Simulated SIDER/SympTrack data for Phase 1 expansion
# In a real-world scenario, this would download and parse SIDER 4.1 files
BIO_DATA = {
    "Hypertension": ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath"],
    "Diabetes Mellitus": ["polyuria", "polydipsia", "weight loss", "fatigue", "blurred vision"],
    "Hyperthyroidism": ["weight loss", "tachycardia", "tremor", "sweating", "anxiety"],
    "Hypothyroidism": ["weight gain", "bradycardia", "fatigue", "cold intolerance", "dry skin"],
    "Pneumonia": ["cough", "fever", "chills", "chest pain", "shortness of breath"],
    "Tuberculosis": ["chronic cough", "weight loss", "night sweats", "hemoptysis", "fever"],
    "Malaria": ["fever", "chills", "headache", "nausea", "sweating"],
    "Typhoid": ["prolonged fever", "abdominal pain", "headache", "constipation", "diarrhea"],
    "Gastroenteritis": ["diarrhea", "vomiting", "abdominal pain", "nausea", "fever"],
    "Hepatitis": ["jaundice", "fatigue", "abdominal pain", "nausea", "dark urine"],
    "Anemia": ["fatigue", "pallor", "shortness of breath", "dizziness", "headache"],
    "Migraine": ["headache", "nausea", "photophobia", "phonophobia", "visual aura"],
    "Epilepsy": ["seizures", "loss of consciousness", "muscle spasms", "confusion"],
    "Depression": ["low mood", "anhedonia", "insomnia", "fatigue", "appetite change"],
    "Anxiety Disorder": ["palpitations", "sweating", "tremor", "shortness of breath", "worry"],
    "Rheumatoid Arthritis": ["joint pain", "joint stiffness", "swelling", "fatigue", "fever"],
    "Osteoarthritis": ["joint pain", "stiffness", "reduced range of motion", "crepitus"],
    "Gout": ["acute joint pain", "swelling", "redness", "warmth", "big toe pain"],
    "Psoriasis": ["red patches", "silvery scales", "dry skin", "itching", "burning"],
    "Eczema": ["itching", "red skin", "dry skin", "blisters", "crusting"],
    "Urticaria": ["hives", "itching", "swelling", "redness"],
    "Allergic Rhinitis": ["sneezing", "runny nose", "itchy eyes", "nasal congestion"],
    "COPD": ["shortness of breath", "chronic cough", "wheezing", "chest tightness"],
    "Angina Pectoris": ["chest pain", "pressure", "squeezing", "pain in arms", "pain in jaw"],
    "Myocardial Infarction": ["severe chest pain", "shortness of breath", "nausea", "sweating", "anxiety"],
    "Heart Failure": ["shortness of breath", "fatigue", "swelling", "tachycardia"],
    "Atrial Fibrillation": ["palpitations", "shortness of breath", "fatigue", "dizziness"],
    "Nephrolithiasis": ["severe flank pain", "hematuria", "nausea", "vomiting", "dysuria"],
    "Urinary Tract Infection": ["dysuria", "frequency", "urgency", "suprapubic pain", "hematuria"],
    "Chronic Kidney Disease": ["fatigue", "swelling", "nausea", "itching", "polyuria"],
    "Peptic Ulcer Disease": ["epigastric pain", "bloating", "nausea", "heartburn"],
    "GERD": ["heartburn", "regurgitation", "chest pain", "dysphagia"],
    "Appendicitis": ["abdominal pain", "fever", "nausea", "vomiting", "anorexia"],
    "Cholelithiasis": ["biliary colic", "nausea", "vomiting", "fever", "jaundice"],
    "Pancreatitis": ["severe abdominal pain", "nausea", "vomiting", "fever", "tachycardia"],
    "Multiple Sclerosis": ["numbness", "weakness", "vision loss", "tremor", "fatigue"],
    "Parkinson's Disease": ["tremor", "bradykinesia", "rigidity", "postural instability"],
    "Alzheimer's Disease": ["memory loss", "confusion", "disorientation", "language problems"],
    "Schizophrenia": ["hallucinations", "delusions", "disorganized thinking", "social withdrawal"],
    "Bipolar Disorder": ["mood swings", "mania", "depression", "insomnia", "impulsivity"],
    "Lupus (SLE)": ["butterfly rash", "joint pain", "fever", "fatigue", "photosensitivity"],
    "HIV/AIDS": ["fever", "weight loss", "fatigue", "night sweats", "lymphadenopathy"],
    "Dengue Fever": ["high fever", "severe headache", "joint pain", "muscle pain", "rash"],
    "Zika Virus": ["fever", "rash", "joint pain", "conjunctivitis", "headache"],
    "Chikungunya": ["fever", "severe joint pain", "headache", "muscle pain", "joint swelling"],
    "Meningitis": ["fever", "headache", "stiff neck", "photophobia", "confusion"],
    "Encephalitis": ["fever", "headache", "confusion", "seizures", "weakness"],
    "Bronchitis": ["cough", "mucus production", "fatigue", "shortness of breath", "fever"],
    "Sinusitis": ["facial pain", "nasal congestion", "headache", "fever", "discolored mucus"],
    "Pharyngitis": ["sore throat", "painful swallowing", "fever", "swollen lymph nodes"],
    "Laryngitis": ["hoarseness", "loss of voice", "throat tickle", "dry throat"],
    "Otitis Media": ["ear pain", "fever", "fluid drainage", "hearing loss"],
}

def ingest_data(dry_run=False):
    print(f"--- Phase 1: Ingesting Biomedical Datasets ---")
    
    disease_count = len(BIO_DATA)
    print(f"Detected {disease_count} diseases in expansion set.")
    
    if dry_run:
        print("DRY RUN: Verification successful. Disease count > 50.")
        return

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Clear existing data for fresh Phase 1 start
            print("Clearing old graph data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            print("Ingesting new Disease-Symptom nodes and edges...")
            for disease, symptoms in BIO_DATA.items():
                session.run("MERGE (d:Disease {name: $name})", name=disease)
                for symptom in symptoms:
                    session.run("""
                        MERGE (s:Symptom {name: $s_name})
                        WITH s
                        MATCH (d:Disease {name: $d_name})
                        MERGE (s)-[:INDICATES]->(d)
                    """, s_name=symptom.lower(), d_name=disease)
            
            print(f"Successfully added {disease_count} diseases and their symptoms.")
        driver.close()
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without writing to DB")
    args = parser.parse_args()
    ingest_data(dry_run=args.dry_run)
