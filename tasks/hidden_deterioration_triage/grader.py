from typing import Any


def _clamp(score: float) -> float:
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return score


def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    detected = sample.get("detected_override", False)
    false_negatives = sample.get("false_negatives", 0)
    base = 1.0 if detected else 0.0
    base -= 0.2 * false_negatives
    return _clamp(base)
