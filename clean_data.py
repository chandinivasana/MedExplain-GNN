import pandas as pd
import os

def clean_medical_dataset():
    input_path = "data/dataset.csv"
    output_path = "data/cleaned_dataset.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load your raw dataset
    df = pd.read_csv(input_path)
    
    # 1. Fix common typos in the Disease column
    corrections = {
        "Peptic ulcer diseae": "Peptic Ulcer Disease",
        "Osteoarthristis": "Osteoarthritis",
        "Dimorphic hemmorhoids(piles)": "Hemorrhoids",
        "Bronchial Asthma": "Asthma"
    }
    df['Disease'] = df['Disease'].replace(corrections)
    
    # 2. Strip whitespace and capitalize disease names
    df['Disease'] = df['Disease'].str.strip().str.title()
    
    # 3. Fix incorrect symptom mappings in the CSV (Data Sanitization)
    # Some diseases have incorrect symptoms like 'high_fever' for 'Varicose Veins'
    # We will remove these to prevent model hallucination.
    
    def sanitize_symptoms(row):
        disease = str(row['Disease'])
        symptoms = [str(s).lower().replace('_', ' ') for s in row[1:].tolist() if pd.notna(s)]
        
        # Define strict clinical definitions to override the noisy dataset
        clinical_definitions = {
            "Type 2 Diabetes": ["increased thirst", "frequent urination", "unintended weight loss", "blurred vision", "fatigue", "slow healing sores"],
            "Diabetes": ["increased thirst", "frequent urination", "unintended weight loss", "blurred vision", "fatigue", "slow healing sores"],
            "Pneumonia": ["high fever", "persistent cough", "chest pain", "shortness of breath", "chills", "fatigue"],
            "Arthritis": ["joint pain", "stiffness", "swelling", "limited range of motion"],
            "Osteoarthritis": ["joint pain", "stiffness", "swelling", "limited range of motion"],
            "Rheumatoid Arthritis": ["joint pain", "stiffness", "swelling", "limited range of motion", "fatigue", "fever"],
            "Hypertension": ["headache", "shortness of breath", "dizziness", "chest pain", "visual changes"],
            "Common Cold": ["runny nose", "congestion", "cough", "sore throat", "fever", "sneezing"]
        }
        
        # General exclusion list (symptoms that should NEVER be associated with certain categories)
        respiratory_symptoms = ["cough", "persistent cough", "congestion", "runny nose", "chest pain"]
        if disease in ["Varicose Veins", "Hemorrhoids", "Arthritis", "Osteoarthritis", "Gout"]:
            symptoms = [s for s in symptoms if s not in respiratory_symptoms]

        # If we have a strict definition, we prioritize it
        if disease in clinical_definitions:
            # We take the intersection of existing symptoms and clinical ones, 
            # OR just use the clinical ones if the row is too noisy.
            # Let's be aggressive to fix the model brain.
            new_symptoms = clinical_definitions[disease]
        else:
            new_symptoms = symptoms

        # Ensure we return a fixed-length list to match the CSV structure
        result = [disease] + new_symptoms
        # Pad with None
        result = result + [None] * (len(row) - len(result))
        return pd.Series(result[:len(row)], index=row.index)

    df = df.apply(sanitize_symptoms, axis=1)
    
    # 4. Drop any rows where the disease name is missing
    df = df.dropna(subset=['Disease'])
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset cleaned successfully! Saved to {output_path}")

if __name__ == "__main__":
    clean_medical_dataset()
