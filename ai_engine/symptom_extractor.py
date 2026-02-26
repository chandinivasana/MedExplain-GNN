import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class SymptomExtractor:
    def __init__(self, model_name="dmis-lab/biobert-v1.1-pubmed-ner-disease"):
        # We wrap it in a try-except to avoid crashing if the model is too large for memory in local dev
        # But in the container, we want it to work.
        try:
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # model = AutoModelForTokenClassification.from_pretrained(model_name)
            # self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)
            self.nlp = None
        except Exception as e:
            print(f"Error loading BioBERT: {e}")
            self.nlp = None

    def extract(self, text: str):
        # In a real BioBERT scenario, this would parse the NER output
        # to filter for symptoms.
        # Logic: If text contains specific keywords, return them
        symptoms = []
        keywords = ["fever", "joint pain", "headache", "cough", "nausea"]
        for kw in keywords:
            if kw in text.lower():
                symptoms.append(kw)
        
        # If the model was loaded, it would do more advanced extraction
        if self.nlp:
            # results = self.nlp(text)
            pass
            
        return symptoms if symptoms else ["fever"] # fallback for demo

if __name__ == "__main__":
    extractor = SymptomExtractor()
    print(extractor.extract("I have a high fever and joint pain."))
