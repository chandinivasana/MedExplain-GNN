import torch
import os

def check_bias():
    graph_path = 'ai_engine/graph_data.pt'
    if not os.path.exists(graph_path):
        print("Graph not found.")
        return
        
    data = torch.load(graph_path)
    disease_indices = range(data.num_symptoms, data.num_symptoms + data.num_classes)
    degrees = torch.zeros(data.num_classes)
    
    for i in range(data.edge_index.size(1)):
        tgt = data.edge_index[1, i].item()
        if tgt in disease_indices:
            degrees[tgt - data.num_symptoms] += 1
            
    # Calculate weights as they are in train.py
    raw_weights = 1.0 / (degrees + 1e-6)
    normalized_weights = raw_weights / raw_weights.sum() * data.num_classes
    
    stats = []
    for i, name in enumerate(data.disease_names):
        stats.append({
            "name": name,
            "degree": degrees[i].item(),
            "weight": normalized_weights[i].item()
        })
        
    # Sort by weight descending
    stats.sort(key=lambda x: x['weight'], reverse=True)
    
    print(f"{'Disease':<35} | {'Degree':<6} | {'Weight':<6}")
    print("-" * 55)
    for s in stats[:15]:
        print(f"{s['name']:<35} | {s['degree']:<6.0f} | {s['weight']:<6.2f}")
    
    print("\n...")
    for s in stats[-5:]:
        print(f"{s['name']:<35} | {s['degree']:<6.0f} | {s['weight']:<6.2f}")

if __name__ == "__main__":
    check_bias()
