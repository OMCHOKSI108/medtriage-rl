from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.medtriage.models import Action, ActionType, BedStatus, BedType, Observation, PatientSummary, Reward, Vitals


VITALS_OVERRIDE = {
    "heart_rate": 100,
    "respiratory_rate": 20,
    "spo2": 92,
}


@dataclass
class Patient:
    patient_id: str
    age: int
    complaint: str
    vitals: Vitals
    resources_expected: int
    requires_immediate: bool = False
    high_acuity: bool = False
    esi_assigned: int | None = None
    stability: float = 1.0
    waiting_steps: int = 0
    assigned_bed: BedType | None = None


@dataclass
class SimState:
    clock: int = 0
    bed_status: BedStatus = field(default_factory=lambda: BedStatus(icu=2, trauma=2, standard=8))
    patients: Dict[str, Patient] = field(default_factory=dict)
    waiting_room: List[str] = field(default_factory=list)


class MedTriageSim:
    def __init__(self) -> None:
        self._state = SimState()

    def reset(self) -> Observation:
        self._state = SimState()
        for index in range(1, 6):
            patient_id = f"P-{index}"
            vitals = Vitals(heart_rate=80 + index, respiratory_rate=16, spo2=98)
            patient = Patient(
                patient_id=patient_id,
                age=30 + index,
                complaint="generalized pain",
                vitals=vitals,
                resources_expected=2,
                requires_immediate=(index == 5),
                high_acuity=(index == 5),
            )
            self._state.patients[patient_id] = patient
            self._state.waiting_room.append(patient_id)
        return self._make_observation()

    def state(self) -> Observation:
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, object]]:
        self._state.clock += 1
        info: Dict[str, object] = {}

        if action.action_type == ActionType.TRIAGE_PATIENT:
            self._apply_triage(action, info)
        elif action.action_type == ActionType.ALLOCATE_BED:
            self._apply_allocate_bed(action, info)
        elif action.action_type == ActionType.ORDER_VITALS_CHECK:
            self._apply_vitals_check(action, info)

        reward = self._advance_patient_states(info)
        done = self._check_done()
        observation = self._make_observation()
        return observation, reward, done, info

    def _apply_triage(self, action: Action, info: Dict[str, object]) -> None:
        if not action.patient_id or action.esi_level is None:
            info["triage_error"] = "missing_patient_or_level"
            return
        patient = self._state.patients.get(action.patient_id)
        if not patient:
            info["triage_error"] = "unknown_patient"
            return
        patient.esi_assigned = action.esi_level
        info["triage_assigned"] = {"patient_id": patient.patient_id, "esi": action.esi_level}

    def _apply_allocate_bed(self, action: Action, info: Dict[str, object]) -> None:
        if not action.patient_id or not action.bed_type:
            info["bed_error"] = "missing_patient_or_bed"
            return
        patient = self._state.patients.get(action.patient_id)
        if not patient:
            info["bed_error"] = "unknown_patient"
            return
        bed_status = self._state.bed_status
        available = getattr(bed_status, action.bed_type.value)
        if available <= 0:
            info["bed_error"] = "no_capacity"
            return
        setattr(bed_status, action.bed_type.value, available - 1)
        patient.assigned_bed = action.bed_type
        if patient.patient_id in self._state.waiting_room:
            self._state.waiting_room.remove(patient.patient_id)
        info["bed_assigned"] = {"patient_id": patient.patient_id, "bed": action.bed_type.value}

    def _apply_vitals_check(self, action: Action, info: Dict[str, object]) -> None:
        if not action.patient_id:
            info["vitals_error"] = "missing_patient"
            return
        patient = self._state.patients.get(action.patient_id)
        if not patient:
            info["vitals_error"] = "unknown_patient"
            return
        info["vitals"] = patient.vitals.model_dump()

    def _advance_patient_states(self, info: Dict[str, object]) -> Reward:
        reward_value = 0.0
        preventable_deaths = 0

        for patient_id in list(self._state.waiting_room):
            patient = self._state.patients[patient_id]
            if patient.high_acuity:
                patient.waiting_steps += 1
                patient.stability -= 0.1

            if self._vitals_override(patient):
                info.setdefault("active_alarms", []).append(patient.patient_id)

            if patient.stability <= 0:
                preventable_deaths += 1
                self._state.waiting_room.remove(patient_id)

        triaged = sum(1 for patient in self._state.patients.values() if patient.esi_assigned is not None)
        bedded = sum(1 for patient in self._state.patients.values() if patient.assigned_bed is not None)

        reward_value += 0.02 * triaged
        reward_value += 0.03 * bedded
        if preventable_deaths > 0:
            reward_value -= 0.5 * preventable_deaths

        info["preventable_deaths"] = preventable_deaths
        components = {
            "triage_progress": 0.02 * triaged,
            "bed_allocation": 0.03 * bedded,
            "preventable_deaths": -0.5 * preventable_deaths,
        }
        return Reward(value=reward_value, components=components)

    def _vitals_override(self, patient: Patient) -> bool:
        return (
            patient.vitals.heart_rate > VITALS_OVERRIDE["heart_rate"]
            or patient.vitals.respiratory_rate > VITALS_OVERRIDE["respiratory_rate"]
            or patient.vitals.spo2 < VITALS_OVERRIDE["spo2"]
        )

    def _check_done(self) -> bool:
        return len(self._state.waiting_room) == 0

    def _make_observation(self) -> Observation:
        waiting_room = [
            PatientSummary(
                patient_id=patient.patient_id,
                age=patient.age,
                complaint=patient.complaint,
                vitals=patient.vitals,
                resources_expected=patient.resources_expected,
                esi_assigned=patient.esi_assigned,
            )
            for patient in (self._state.patients[pid] for pid in self._state.waiting_room)
        ]
        active_alarms = [
            patient.patient_id
            for patient in self._state.patients.values()
            if self._vitals_override(patient)
        ]
        return Observation(
            waiting_room=waiting_room,
            bed_status=self._state.bed_status,
            active_alarms=active_alarms,
            simulation_clock=self._state.clock,
        )
