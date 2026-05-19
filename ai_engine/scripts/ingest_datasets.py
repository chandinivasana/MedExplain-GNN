import os
from neo4j import GraphDatabase

REAL_DISEASE_DATA = {
    "Diabetes": {
        "symptoms": ["excessive thirst", "frequent urination", "blurred vision", "fatigue"],
        "foods_to_avoid": ["Sugary Foods", "White Bread", "Sweetened Cereals"]
    },
    "Hypertension": {
        "symptoms": ["severe headache", "chest pain", "dizziness", "shortness of breath"],
        "foods_to_avoid": ["Salty Foods", "Processed Meats", "Canned Soups"]
    },
    "Malaria": {
        "symptoms": ["high fever", "chills", "sweating", "muscle pain"],
        "foods_to_avoid": ["Oily Foods", "Spicy Foods"]
    },
    "Tuberculosis": {
        "symptoms": ["persistent cough", "chest pain", "weight loss", "night sweats"],
        "foods_to_avoid": ["Alcohol", "Refined Carbohydrates"]
    },
    "Pneumonia": {
        "symptoms": ["cough with phlegm", "fever", "chills", "difficulty breathing"],
        "foods_to_avoid": ["Cold Dairy", "Excessive Salt"]
    },
    "Anemia": {
        "symptoms": ["fatigue", "pale skin", "shortness of breath", "cold hands"],
        "foods_to_avoid": ["Tea with Meals", "Coffee with Meals"]
    },
    "Arthritis": {
        "symptoms": ["joint pain", "stiffness", "swelling", "reduced range of motion"],
        "foods_to_avoid": ["Refined Sugars", "Saturated Fats", "MSG"]
    },
    "Bronchitis": {
        "symptoms": ["productive cough", "fatigue", "shortness of breath", "slight fever"],
        "foods_to_avoid": ["Mucus-Producing Dairy", "Sugary Drinks"]
    },
    "Cholera": {
        "symptoms": ["watery diarrhea", "nausea", "vomiting", "dehydration"],
        "foods_to_avoid": ["Raw Seafood", "Unpeeled Vegetables"]
    },
    "Chickenpox": {
        "symptoms": ["itchy rash", "fever", "headache", "loss of appetite"],
        "foods_to_avoid": ["Acidic Foods", "Salty Foods"]
    },
    "Gout": {
        "symptoms": ["intense joint pain", "redness", "swelling", "heat in joints"],
        "foods_to_avoid": ["Red Meat", "Shellfish", "Beer"]
    },
    "Typhoid": {
        "symptoms": ["prolonged fever", "fatigue", "headache", "abdominal pain"],
        "foods_to_avoid": ["Raw Vegetables", "Unpasteurized Milk"]
    },
    "Gastritis": {
        "symptoms": ["burning stomach pain", "nausea", "vomiting", "fullness after eating"],
        "foods_to_avoid": ["Spicy Foods", "Fried Foods", "Acidic Fruit"]
    },
    "Hepatitis": {
        "symptoms": ["jaundice", "fatigue", "dark urine", "nausea"],
        "foods_to_avoid": ["Alcohol", "Raw Shellfish", "Fried Foods"]
    },
    "Kidney Stones": {
        "symptoms": ["severe back pain", "blood in urine", "nausea", "frequent urination"],
        "foods_to_avoid": ["High-Oxalate Foods", "Spinach", "Rhubarb"]
    },
    "Measles": {
        "symptoms": ["fever", "dry cough", "runny nose", "skin rash"],
        "foods_to_avoid": ["Dairy Products", "Sugary Foods"]
    },
    "Meningitis": {
        "symptoms": ["stiff neck", "high fever", "headache", "seizures"],
        "foods_to_avoid": ["Processed Sugars", "Trans Fats"]
    },
    "Osteoporosis": {
        "symptoms": ["back pain", "loss of height", "stooped posture", "easy fractures"],
        "foods_to_avoid": ["Excessive Alcohol", "Soda", "High Caffeine"]
    },
    "Psoriasis": {
        "symptoms": ["red patches of skin", "dry cracked skin", "itching", "swollen joints"],
        "foods_to_avoid": ["Alcohol", "Dairy Products", "Nightshades"]
    },
    "Tonsillitis": {
        "symptoms": ["sore throat", "difficulty swallowing", "swollen tonsils", "fever"],
        "foods_to_avoid": ["Crunchy Foods", "Acidic Juices"]
    },
    "Ulcer": {
        "symptoms": ["burning stomach pain", "bloating", "heartburn", "nausea"],
        "foods_to_avoid": ["Spicy Pepper", "Caffeine", "Citrus"]
    },
    "Alzheimer": {
        "symptoms": ["memory loss", "confusion", "difficulty planning", "mood changes"],
        "foods_to_avoid": ["Trans Fats", "Highly Processed Foods"]
    },
    "Parkinson": {
        "symptoms": ["tremors", "slowed movement", "rigid muscles", "impaired posture"],
        "foods_to_avoid": ["High Protein at Night", "Dairy"]
    },
    "Epilepsy": {
        "symptoms": ["temporary confusion", "staring spell", "uncontrollable jerking", "loss of consciousness"],
        "foods_to_avoid": ["High Sugar", "MSG", "Caffeine"]
    },
    "Sinusitis": {
        "symptoms": ["facial pain", "stuffy nose", "headache", "sore throat"],
        "foods_to_avoid": ["Cold Milk", "Refined Sugars"]
    },
    "Appendicitis": {
        "symptoms": ["sudden pain in right abdomen", "nausea", "vomiting", "low fever"],
        "foods_to_avoid": ["High Fat Foods", "Sugary Drinks"]
    },
    "Cirrhosis": {
        "symptoms": ["fatigue", "easy bleeding", "jaundice", "swelling in legs"],
        "foods_to_avoid": ["Alcohol", "Raw Shellfish", "High Salt"]
    },
    "Insomnia": {
        "symptoms": ["difficulty falling asleep", "waking up during night", "daytime sleepiness", "irritability"],
        "foods_to_avoid": ["Late Night Caffeine", "Heavy Spicy Meals"]
    },
    "Lupus": {
        "symptoms": ["butterfly rash", "joint pain", "fatigue", "sun sensitivity"],
        "foods_to_avoid": ["Alfalfa Sprouts", "Garlic", "Highly Processed Foods"]
    },
    "Lyme Disease": {
        "symptoms": ["bullseye rash", "fever", "chills", "fatigue"],
        "foods_to_avoid": ["Inflammatory Sugars", "Dairy"]
    },
    "Mumps": {
        "symptoms": ["swollen salivary glands", "fever", "headache", "muscle aches"],
        "foods_to_avoid": ["Acidic Fruits", "Hard-to-Chew Foods"]
    },
    "Scurvy": {
        "symptoms": ["bleeding gums", "joint pain", "bruising", "fatigue"],
        "foods_to_avoid": ["Canned Goods with No Vitamin C"]
    },
    "Tetanus": {
        "symptoms": ["jaw cramping", "muscle spasms", "stiffness in neck", "fever"],
        "foods_to_avoid": ["Difficult to Swallow Foods"]
    },
    "Dengue": {
        "symptoms": ["high fever", "joint pain", "headache", "nausea", "pain behind eyes"],
        "foods_to_avoid": ["Salty Foods", "Oily Foods"]
    },
    "Influenza": {
        "symptoms": ["fever", "headache", "cough", "nausea", "fatigue"],
        "foods_to_avoid": ["Sugary Foods", "Alcohol"]
    },
    "Asthma": {
        "symptoms": ["cough", "shortness of breath", "wheezing", "chest tightness"],
        "foods_to_avoid": ["Processed Foods", "Sulphites", "Cold Dairy"]
    },
    "Migraine": {
        "symptoms": ["headache", "nausea", "sensitivity to light", "sensitivity to sound"],
        "foods_to_avoid": ["Aged Cheese", "Chocolate", "Red Wine"]
    },
    "Common Cold": {
        "symptoms": ["cough", "sore throat", "runny nose", "sneezing"],
        "foods_to_avoid": ["Iced Drinks", "Cold Milk"]
    },
    "Eczema": {
        "symptoms": ["dry skin", "itching", "red patches", "small bumps"],
        "foods_to_avoid": ["Cow's Milk", "Eggs", "Soy Products"]
    },
    "Anxiety": {
        "symptoms": ["restlessness", "racing heart", "rapid breathing", "sweating"],
        "foods_to_avoid": ["Excessive Caffeine", "Alcohol", "High Sugar"]
    },
    "Cataract": {
        "symptoms": ["cloudy vision", "difficulty seeing at night", "halos around lights"],
        "foods_to_avoid": ["High Sugar", "Trans Fats"]
    },
    "Conjunctivitis": {
        "symptoms": ["redness in eye", "itchiness", "gritty feeling", "discharge"],
        "foods_to_avoid": ["Inflammatory Foods"]
    },
    "Dermatitis": {
        "symptoms": ["itchy rash", "dry skin", "blisters", "swelling"],
        "foods_to_avoid": ["Dairy", "Gluten", "Eggs"]
    },
    "Glaucoma": {
        "symptoms": ["patchy blind spots", "tunnel vision", "severe headache", "eye pain"],
        "foods_to_avoid": ["High Caffeine", "Trans Fats"]
    },
    "Heart Failure": {
        "symptoms": ["shortness of breath", "fatigue", "swelling in legs", "rapid heartbeat"],
        "foods_to_avoid": ["Excessive Salt", "Saturated Fats"]
    },
    "Hyperthyroidism": {
        "symptoms": ["weight loss", "rapid heartbeat", "increased appetite", "nervousness"],
        "foods_to_avoid": ["Excess Iodine", "Caffeine"]
    },
    "Hypothyroidism": {
        "symptoms": ["fatigue", "weight gain", "cold sensitivity", "dry skin"],
        "foods_to_avoid": ["Soy", "Cruciferous Vegetables (Raw)", "Gluten"]
    },
    "Jaundice": {
        "symptoms": ["yellow skin", "yellow eyes", "pale stools", "dark urine"],
        "foods_to_avoid": ["Alcohol", "Fried Foods", "Excessive Salt"]
    },
    "Laryngitis": {
        "symptoms": ["hoarseness", "weak voice", "sore throat", "dry cough"],
        "foods_to_avoid": ["Caffeine", "Alcohol", "Spicy Foods"]
    },
    "Leukemia": {
        "symptoms": ["fever", "persistent fatigue", "easy bleeding", "frequent infections"],
        "foods_to_avoid": ["Raw Unpasteurized Foods"]
    },
    "Narcolepsy": {
        "symptoms": ["sudden sleep attacks", "loss of muscle tone", "sleep paralysis", "hallucinations"],
        "foods_to_avoid": ["Heavy Carbohydrates", "Caffeine before naps"]
    },
    "Obesity": {
        "symptoms": ["increased body fat", "shortness of breath", "sweating", "snoring"],
        "foods_to_avoid": ["Ultra-Processed Foods", "Sugary Drinks", "Fast Food"]
    },
    "Osteoarthritis": {
        "symptoms": ["joint pain", "stiffness", "tenderness", "loss of flexibility"],
        "foods_to_avoid": ["Added Sugars", "Saturated Fats"]
    },
    "Rheumatoid Arthritis": {
        "symptoms": ["tender warm joints", "joint stiffness", "fatigue", "fever"],
        "foods_to_avoid": ["Processed Meats", "Refined Sugars"]
    },
    "Sleep Apnea": {
        "symptoms": ["loud snoring", "episodes of stopped breathing", "abrupt awakenings", "morning headache"],
        "foods_to_avoid": ["Alcohol", "High-Fat Meals before sleep"]
    },
    "Stroke": {
        "symptoms": ["trouble speaking", "paralysis in face/arm", "blurred vision", "sudden headache"],
        "foods_to_avoid": ["High Sodium", "Trans Fats", "Sugary Drinks"]
    }
}

def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for disease, data in REAL_DISEASE_DATA.items():
            # Create Disease, Symptoms, Foods and all relationships in one go for atomicity
            query = """
            MERGE (d:Disease {name: $d_name})
            WITH d
            UNWIND $symptoms AS s_name
            MERGE (s:Symptom {name: s_name})
            MERGE (s)-[:INDICATES]->(d)
            WITH d
            UNWIND $foods AS f_name
            MERGE (f:Food {name: f_name})
            MERGE (d)-[:CONTRAINDICATED]->(f)
            """
            session.run(query, d_name=disease, symptoms=data["symptoms"], foods=data["foods_to_avoid"])
            print(f"Ingested {disease}")
            
    driver.close()
    print("Done.")

if __name__ == "__main__":
    main()
