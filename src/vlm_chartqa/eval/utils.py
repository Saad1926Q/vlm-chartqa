def _to_float(text: str):
    try:
        if text.endswith("%"):  # Convert percentage to float
            return float(text.rstrip("%")) / 100.0
        else:
            return float(text)
    except ValueError:
        return None

def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """
    For numeric answers: relaxed correctness (within 5% relative change).
    For non-numeric answers: exact match (case-insensitive).
    Adapted from EvolvingLMMs-Lab/lmms-eval.
    """
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)

    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
