from typing import Any


def _clamp(score: float) -> float:
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return score


def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    correct_esi = sample.get("correct_esi_count", 0)
    total_patients = item.get("total_patients", 15)
    preventable_deaths = sample.get("preventable_deaths", 0)

    accuracy = correct_esi / max(total_patients, 1)
    if preventable_deaths > 0:
        accuracy -= 0.3 * preventable_deaths
    return _clamp(accuracy)
