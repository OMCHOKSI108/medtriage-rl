def grade(sim_state: dict) -> float:
    try:
        esi1_in_icu = sim_state.get("esi1_patients_in_icu", 0)
        total_esi1 = sim_state.get("total_esi1_patients", 1)
        deaths = sim_state.get("preventable_deaths", 0)
        total = sim_state.get("total_patients", 15)
        icu_ratio = esi1_in_icu / total_esi1 if total_esi1 > 0 else 0.0
        death_ratio = 1.0 - (deaths / total) if total > 0 else 0.0
        score = 0.6 * icu_ratio + 0.4 * death_ratio
        return float(min(max(score, 0.01), 0.99))
    except Exception:
        return 0.01
