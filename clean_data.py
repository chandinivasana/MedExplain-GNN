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
    
    # 3. Drop any rows where the disease name is missing
    df = df.dropna(subset=['Disease'])
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset cleaned successfully! Saved to {output_path}")

if __name__ == "__main__":
    clean_medical_dataset()
