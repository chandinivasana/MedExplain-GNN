import torch
import torch.nn.functional as F
import os
import sys
from collections import Counter

# Add ai_engine to path to import the model class
sys.path.append(os.path.join(os.getcwd(), 'ai_engine'))
from model import MedicalGAT

def diagnostic_report():
    print("="*60)
    print("      LEAD DATA SCIENTIST AUDIT: GAT MODEL DIAGNOSTICS")
    print("="*60)

    # 1. Load Data and Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    graph_path = 'ai_engine/graph_data.pt'
    model_path = 'ai_engine/best_model.pth'

    if not os.path.exists(graph_path) or not os.path.exists(model_path):
        print("Error: Required files (graph_data.pt or best_model.pth) not found.")
        return

    data = torch.load(graph_path, map_location=device)
    num_node_features = data.x.size(1)
    num_classes = data.num_classes
    num_symptoms = data.num_symptoms

    model = MedicalGAT(num_node_features=num_node_features, hidden_channels=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 1. THE VOCABULARY REPORT ---
    print(f"\n[1] VOCABULARY REPORT")
    print(f"Total Unique Diseases learned: {num_classes}")
    print(f"Total Unique Symptoms in graph: {num_symptoms}")
    
    print("\nTop 20 Learned Diseases:")
    for i, name in enumerate(data.disease_names[:20]):
        print(f" {i+1:2d}. {name}")

    # --- 2. CLASS IMBALANCE CHECK ---
    print(f"\n[2] CLASS IMBALANCE CHECK")
    # In this node-classification setup, we check the degree of each disease node
    # (how many symptoms are connected to it)
    edge_index = data.edge_index.cpu()
    # Disease indices start from num_symptoms
    disease_indices = range(num_symptoms, num_symptoms + num_classes)
    
    connections = Counter()
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()
        if tgt in disease_indices:
            connections[data.disease_names[tgt - num_symptoms]] += 1

    most_common = connections.most_common(5)
    print("Diseases with most symptom connections (Potential Bias):")
    for name, count in most_common:
        print(f" - {name}: {count} symptoms")

    # Check for specific "Vertigo" issue mentioned by user
    vertigo_count = connections.get("Vertigo", 0)
    if vertigo_count > 0:
        avg_conn = sum(connections.values()) / len(connections)
        if vertigo_count > avg_conn * 1.5:
            print(f"ALERT: 'Vertigo' is over-represented with {vertigo_count} connections (Avg: {avg_conn:.1f})")
    else:
        # Check if it's under a different name due to cleaning
        for name in connections:
            if "Vertigo" in name:
                print(f"Note: Found '{name}' with {connections[name]} connections.")

    # --- 3. THE SANITY CHECK: DUMMY PATIENTS ---
    print(f"\n[3] SANITY CHECK (INFERENCE AUDIT)")
    
    symptom_map = {name.lower(): i for i, name in enumerate(data.symptom_names)}
    
    patients = [
        {"name": "Patient A (Respiratory)", "symptoms": ["fever", "cough"]},
        {"name": "Patient B (Joint/Autoimmune)", "symptoms": ["joint pain", "swelling"]},
        {"name": "Patient C (Vertigo/Nausea)", "symptoms": ["vomiting", "headache", "nausea"]}
    ]

    for patient in patients:
        print(f"\nProcessing {patient['name']}...")
        x_input = data.x.clone()
        active = False
        for s in patient['symptoms']:
            if s in symptom_map:
                idx = symptom_map[s]
                x_input[idx] *= 2.0 # Simulate activation boost
                active = True
        
        if not active:
            print(" Skipping: No symptoms matched graph.")
            continue

        with torch.no_grad():
            logits = model(x_input, data.edge_index)
            disease_logits = logits[num_symptoms:]
            
            # Use Temperature=1.0 for raw audit
            probs = F.softmax(disease_logits.diag(), dim=0)
            
            top_val, top_idx = torch.topk(probs, 3)
            
            print(f" Raw Top 3 Probabilities:")
            for i in range(3):
                d_name = data.disease_names[top_idx[i].item()]
                print(f"  - {d_name}: {top_val[i].item():.6f}")

            # Statistical check for model collapse
            std_dev = torch.std(probs).item()
            if std_dev < 1e-5:
                print(" WARNING: Model Collapse detected! The probability distribution is flat.")

    print("\n" + "="*60)
    print("      DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    diagnostic_report()
