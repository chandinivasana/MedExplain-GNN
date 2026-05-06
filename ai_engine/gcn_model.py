import warnings

try:
    from .inference import predict_disease_from_symptoms
except ImportError:
    from inference import predict_disease_from_symptoms


def _fallback_prediction(symptom_nodes):
    if "joint pain" in symptom_nodes:
        return "Dengue", 0.92
    return "Influenza", 0.85


def get_disease_prediction(symptom_nodes):
    try:
        disease, confidence, _top_predictions = predict_disease_from_symptoms(symptom_nodes)
        return disease, confidence
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        warnings.warn(
            "Falling back to demo prediction because trained model inference "
            f"is not available: {exc}",
            RuntimeWarning,
        )
        return _fallback_prediction(symptom_nodes)


if __name__ == "__main__":
    print(get_disease_prediction(["fever", "joint pain"]))
