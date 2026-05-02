import pandas as pd
import sys

def find_best_match(symptoms):
    df = pd.read_csv('data/cleaned_dataset.csv')
    
    results = []
    for disease in df['Disease'].unique():
        sub = df[df['Disease'] == disease]
        all_syms = set()
        for r in sub.values:
            for s in r[1:]:
                if pd.notna(s):
                    all_syms.add(str(s).lower().replace('_', ' '))
        
        matches = [s for s in symptoms if s in all_syms]
        results.append({
            "disease": disease,
            "match_count": len(matches),
            "matches": matches,
            "total_symptoms": len(all_syms)
        })
        
    results.sort(key=lambda x: x['match_count'], reverse=True)
    
    print(f"Top 5 Data Matches for {symptoms}:")
    for r in results[:10]:
        print(f" - {r['disease']}: {r['match_count']} matches ({r['matches']}) [Total Syms: {r['total_symptoms']}]")

if __name__ == "__main__":
    test_symptoms = ['high fever', 'persistent cough', 'chest pain']
    find_best_match(test_symptoms)
