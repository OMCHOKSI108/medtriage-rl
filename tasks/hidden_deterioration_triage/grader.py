def grade(sim_state: dict) -> float:
    try:
        hidden_escalated = sim_state.get("hidden_patient_escalated", False)
        other_triaged = sim_state.get("other_patients_triaged", 0)
        total_other = sim_state.get("total_other_patients", 4)
        if not hidden_escalated:
            return 0.01
        partial = (other_triaged / total_other) if total_other > 0 else 0.0
        score = 0.5 + 0.5 * partial
        return float(min(max(score, 0.01), 0.99))
    except Exception:
        return 0.01
