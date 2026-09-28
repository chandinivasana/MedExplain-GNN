"""
MedExplain-GNN: Interactive Production Demo (Hugging Face Spaces & Local)
---
title: MedExplain GNN
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: demo_app.py
pinned: false
license: mit
---
"""

import sys
import os
import gradio as gr

# Add ai_engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_engine'))
from symptom_extractor import SymptomExtractor
from inference import GATInferenceEngine

print("Initializing Symptom Extractor and GAT Inference Engine...")
extractor = SymptomExtractor()
engine = GATInferenceEngine()
print("Engines initialized successfully!")

def analyze_symptoms(text):
    if not text or not text.strip():
        return (
            {}, 
            "Please enter symptom descriptions or click one of the clinical examples below.",
            "N/A",
            "N/A"
        )
    
    # 1. Extraction via BioBERT NER / Clinical Parser
    extracted_symptoms = extractor.extract(text)
    if not extracted_symptoms:
        return (
            {"No Matching Clinical Symptoms": 0.0},
            "No recognizable medical symptoms were identified in the input description.",
            "N/A",
            "N/A"
        )
    
    # 2. GAT Model Inference
    predictions, dietary_precautions, cypher_query, attention_weights = engine.predict(extracted_symptoms)
    
    # Format label scores for Gradio Label component
    pred_dict = {disease: float(prob) for disease, prob in predictions}
    
    # Format Explanation
    top_disease, top_conf = predictions[0]
    explanation = (
        f"**Primary Diagnosis:** `{top_disease}` ({top_conf*100:.1f}% confidence)\n\n"
        f"**Extracted Symptoms:** {', '.join([f'`{s}`' for s in extracted_symptoms])}\n\n"
        f"**Graph Reasoning:** Multi-head attention across the bipartite symptom-disease graph "
        f"identified `{top_disease}` as the strongest latent attractor node after megaphone feature scaling."
    )
    
    # Format Dietary Rules
    if dietary_precautions:
        dietary_str = "\n".join([f"- {item}" for item in dietary_precautions])
    else:
        dietary_str = "No specific contraindications recorded."
        
    return pred_dict, explanation, cypher_query, dietary_str

# Custom theme & CSS
custom_css = """
.container { max-width: 1000px; margin: auto; }
.output-box { border-radius: 12px; }
"""

with gr.Blocks(title="MedExplain-GNN: Explainable Medical AI") as demo:
    gr.Markdown(
        """
        # 🧬 MedExplain-GNN: Explainable Medical Reasoning Engine
        ### Graph Attention Networks (GAT) + BioBERT Semantic Embeddings + Neo4j Knowledge Graph
        Translates unstructured patient symptom descriptions into calibrated differential diagnoses with transparent graph justifications and dietary contraindications.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Clinical Symptom Description",
                placeholder="Enter freeform patient symptoms (e.g., patient complains of high fever, dry cough, loss of taste, and mild fatigue)...",
                lines=4
            )
            analyze_btn = gr.Button("🔬 Run GNN Diagnostic Inference", variant="primary")
            
            gr.Examples(
                examples=[
                    ["high fever, dry cough, loss of taste, fatigue, headache"],
                    ["joint pain, swelling, morning stiffness, fatigue"],
                    ["severe headache, sensitivity to light, nausea, dizziness"],
                    ["increased thirst, frequent urination, blurred vision, weight loss"],
                    ["abdominal cramps, watery diarrhea, low grade fever, nausea"]
                ],
                inputs=[input_text],
                label="Clinical Preset Scenarios"
            )
            
        with gr.Column(scale=1):
            predictions_output = gr.Label(num_top_classes=3, label="Calibrated Differential Diagnosis (GAT)")
            explanation_output = gr.Markdown(label="Clinical Reasoning Path")
            cypher_output = gr.Code(label="Explainability Layer: Neo4j Cypher Query", language="sql")
            dietary_output = gr.Markdown(label="Dietary Precautions & Contraindications")
            
    analyze_btn.click(
        fn=analyze_symptoms,
        inputs=[input_text],
        outputs=[predictions_output, explanation_output, cypher_output, dietary_output]
    )
    
    gr.Markdown(
        """
        ---
        **⚠️ Research & Educational Disclaimer:** MedExplain-GNN is an AI research prototype designed to evaluate Graph Attention Networks and Explainable AI (XAI) in clinical informatics. It is **not** an FDA-cleared Software as a Medical Device (SaMD) and should not be used as a substitute for professional medical consultation, diagnosis, or treatment.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
