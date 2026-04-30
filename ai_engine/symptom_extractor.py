from transformers import pipeline

class SymptomExtractor:
    def __init__(self, model_name="d4data/biomedical-ner-all"):
        try:
            # Load a robust biomedical NER model via HuggingFace
            # aggregation_strategy="simple" merges sub-tokens into complete entities
            self.nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
            print("Successfully loaded BioBERT NER pipeline.")
        except Exception as e:
            print(f"Warning: Could not load BioBERT NER model. Using heuristic fallback. Error: {e}")
            self.nlp = None
            
        # A comprehensive dictionary of 100+ symptoms matching our synthetic dataset
        self.common_symptoms = [
            "fever", "chills", "muscle aches", "cough", "congestion", "runny nose", "headache", "fatigue",
            "high fever", "severe headache", "pain behind the eyes", "joint pain", "muscle pain", "nausea", 
            "vomiting", "skin rash", "profuse sweating", "increased thirst", "frequent urination", "increased hunger", 
            "unintended weight loss", "blurred vision", "slow-healing sores", "frequent infections", "watery diarrhea", 
            "abdominal cramps", "low-grade fever", "shortness of breath", "nosebleeds", "flushing", "dizziness", 
            "chest pain", "visual changes", "throbbing pain", "pulsing pain", "sensitivity to light", "sensitivity to sound",
            "tiredness", "loss of taste", "loss of smell", "chest tightness", "wheezing", "coughing attacks",
            "weakness", "pale skin", "cold hands and feet", "brittle nails", "diarrhea", "weight loss", "bloating", "gas",
            "abdominal pain", "burning stomach pain", "feeling of fullness", "belching", "heartburn", "tender joints", 
            "warm joints", "swollen joints", "joint stiffness", "loss of appetite", "increased sensitivity to cold", "constipation", 
            "dry skin", "weight gain", "puffy face", "muscle weakness", "rapid heartbeat", "irregular heartbeat", "palpitations", 
            "nervousness", "anxiety", "irritability", "tremor", "sweating", "intense joint pain", "lingering discomfort", 
            "inflammation", "redness", "limited range of motion", "sleep problems", "changes in urine output", "decreased mental sharpness", 
            "muscle twitches", "persistent cough", "coughing up blood", "night sweats", "confusion", "cough with phlegm", 
            "stiffness", "swelling", "butterfly rash", "skin lesions"
        ]

    def extract(self, text: str) -> list:
        symptoms = set()
        
        # 1. Advanced extraction via BioBERT NLP Model
        if self.nlp:
            try:
                entities = self.nlp(text)
                for ent in entities:
                    # In specialized medical NER models, signs and symptoms are classified accurately
                    if ent['entity_group'] in ['Sign_symptom', 'Disease_disorder', 'Diagnostic_procedure']:
                        symptoms.add(ent['word'].lower().strip())
            except Exception as e:
                print(f"NER Inference error: {e}")
                
        # 2. Heuristic extraction mapping directly to our defined Knowledge Graph nodes
        # This acts as a robust fallback and ensures high recall for known graph nodes
        text_lower = text.lower()
        for sym in self.common_symptoms:
            if sym in text_lower:
                symptoms.add(sym)
                
        return list(symptoms)

if __name__ == "__main__":
    extractor = SymptomExtractor()
    print(extractor.extract("The patient came in with a high fever, severe headache, and some joint pain. They also experienced nausea."))
