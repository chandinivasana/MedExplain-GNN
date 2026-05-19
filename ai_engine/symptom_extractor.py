from transformers import pipeline

try:
    import fitz
except ImportError:
    fitz = None


class SymptomExtractor:
    def __init__(self, model_name="d4data/biomedical-ner-all"):
        try:
            self.nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
            print("Successfully loaded BioBERT NER pipeline.")
        except Exception as exc:
            print(f"Warning: Could not load BioBERT NER model. Using heuristic fallback. Error: {exc}")
            self.nlp = None

        self.common_symptoms = {
            "fever": "fever",
            "high fever": "fever",
            "chills": "chills",
            "muscle pain": "muscle pain",
            "muscle aches": "muscle pain",
            "cough": "cough",
            "congestion": "congestion",
            "runny nose": "runny nose",
            "headache": "headache",
            "severe headache": "headache",
            "fatigue": "fatigue",
            "joint pain": "joint pain",
            "severe joint pain": "joint pain",
            "pain behind eyes": "pain behind eyes",
            "pain behind my eyes": "pain behind eyes",
            "pain behind the eyes": "pain behind eyes",
            "nausea": "nausea",
            "vomiting": "vomiting",
            "diarrhea": "diarrhea",
            "abdominal pain": "abdominal pain",
            "abdominal cramps": "abdominal pain",
            "shortness of breath": "shortness of breath",
            "wheezing": "wheezing",
            "chest tightness": "chest tightness",
            "sore throat": "sore throat",
            "sensitivity to light": "sensitivity to light",
            "sensitivity to sound": "sensitivity to sound",
        }

    def extract(self, text: str) -> list:
        symptoms = set()

        if self.nlp:
            try:
                entities = self.nlp(text)
                for entity in entities:
                    if entity.get("entity_group") in {
                        "Sign_symptom",
                        "Disease_disorder",
                        "Diagnostic_procedure",
                    }:
                        word = entity["word"].replace("##", "").lower().strip()
                        if len(word) > 1:
                            symptoms.add(word)
            except Exception as exc:
                print(f"NER inference error: {exc}")

        text_lower = text.lower().replace("_", " ")
        for keyword, symptom in self.common_symptoms.items():
            if keyword in text_lower:
                symptoms.add(symptom)

        return sorted(symptoms)

    def extract_from_pdf(self, pdf_content: bytes) -> list:
        if fitz is None:
            raise RuntimeError("PyMuPDF is not installed.")

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return self.extract(full_text)
