def grade(sim_state: dict) -> float:
    try:
        triaged = sim_state.get("triaged_count", 0)
        total = sim_state.get("total_patients", 5)
        correct_esi = sim_state.get("correct_esi_count", 0)
        deaths = sim_state.get("preventable_deaths", 0)
        if total == 0:
            return 0.01
        accuracy = correct_esi / total
        death_penalty = 0.3 * deaths
        score = accuracy - death_penalty
        return float(min(max(score, 0.01), 0.99))
    except Exception:
        return 0.01
