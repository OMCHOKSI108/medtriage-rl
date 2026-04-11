from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from models import Action, ActionType, BedStatus, BedType, Observation, PatientSummary, Reward, Vitals


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
        self._rng = random.Random(42)
        self.current_task_id = "routine_resource_allocation"
        self.simulation_clock = 0
        self.preventable_deaths = 0
        self.waiting_room: List[Patient] = []
        self.active_alarms: List[str] = []
        self.bed_status = {"icu": 4, "trauma": 2, "standard": 10}

    def reset(self, task_id: str = "routine_resource_allocation", seed: int = 42) -> Observation:
        self._rng = random.Random(seed)
        self.current_task_id = task_id
        self.simulation_clock = 0
        self.preventable_deaths = 0

        if task_id == "routine_resource_allocation":
            patients = self._gen_routine_patients()
            self.bed_status = {"icu": 4, "trauma": 2, "standard": 10}
        elif task_id == "hidden_deterioration_triage":
            patients = self._gen_hidden_deterioration_patients()
            self.bed_status = {"icu": 4, "trauma": 2, "standard": 10}
        elif task_id == "mass_casualty_surge":
            patients = self._gen_mass_casualty_patients()
            self.bed_status = {"icu": 2, "trauma": 2, "standard": 6}
        else:
            patients = self._gen_routine_patients()
            self.bed_status = {"icu": 4, "trauma": 2, "standard": 10}

        self.waiting_room = patients
        self.active_alarms = self._compute_active_alarms(patients)

        self._state = SimState(
            clock=0,
            bed_status=BedStatus(**self.bed_status),
            patients={p.patient_id: p for p in patients},
            waiting_room=[p.patient_id for p in patients],
        )
        return self._make_observation()

    def _gen_routine_patients(self) -> List[Patient]:
        complaints = [
            "ankle sprain",
            "mild headache",
            "skin rash",
            "sore throat",
            "low back pain",
        ]
        patients: List[Patient] = []
        for index in range(5):
            patient = Patient(
                patient_id=f"P_ROUTINE_{index + 1}",
                age=self._rng.randint(20, 75),
                complaint=complaints[index],
                vitals=Vitals(
                    heart_rate=self._rng.randint(65, 85),
                    respiratory_rate=self._rng.randint(14, 18),
                    spo2=self._rng.randint(96, 99),
                ),
                resources_expected=1,
                requires_immediate=False,
                high_acuity=False,
                esi_assigned=self._rng.randint(3, 5),
            )
            patients.append(patient)
        return patients

    def _gen_hidden_deterioration_patients(self) -> List[Patient]:
        benign_complaints = [
            "mild cough",
            "seasonal allergies",
            "tension headache",
            "minor nausea",
        ]
        patients: List[Patient] = []
        for index in range(4):
            patient = Patient(
                patient_id=f"P_BENIGN_{index + 1}",
                age=self._rng.randint(18, 70),
                complaint=benign_complaints[index],
                vitals=Vitals(
                    heart_rate=self._rng.randint(70, 90),
                    respiratory_rate=self._rng.randint(14, 18),
                    spo2=self._rng.randint(95, 99),
                ),
                resources_expected=1,
                requires_immediate=False,
                high_acuity=False,
                esi_assigned=self._rng.randint(3, 5),
            )
            patients.append(patient)

        patients.append(
            Patient(
                patient_id="P_HIDDEN",
                age=self._rng.randint(25, 65),
                complaint="mild abdominal discomfort",
                vitals=Vitals(heart_rate=114, respiratory_rate=23, spo2=88),
                resources_expected=2,
                requires_immediate=True,
                high_acuity=True,
                esi_assigned=2,
            )
        )
        return patients

    def _gen_mass_casualty_patients(self) -> List[Patient]:
        injuries = [
            "blunt trauma",
            "smoke inhalation",
            "cardiac chest pain",
            "open fracture",
            "burn injury",
            "head trauma",
            "pelvic fracture",
            "thoracic trauma",
            "limb crush injury",
            "facial burns",
            "abdominal trauma",
            "spinal tenderness",
            "ankle fracture",
            "arm laceration",
            "suspected arrhythmia",
        ]
        patients: List[Patient] = []
        low_spo2_indices = {0, 3, 7, 10}
        for index in range(15):
            is_critical = index < 5
            spo2 = self._rng.randint(84, 91) if index in low_spo2_indices else self._rng.randint(92, 98)
            patient = Patient(
                patient_id=f"P_MASS_{index + 1}",
                age=self._rng.randint(16, 85),
                complaint=injuries[index],
                vitals=Vitals(
                    heart_rate=self._rng.randint(105, 140) if is_critical else self._rng.randint(85, 110),
                    respiratory_rate=self._rng.randint(22, 30) if is_critical else self._rng.randint(16, 22),
                    spo2=spo2,
                ),
                resources_expected=self._rng.randint(1, 3),
                requires_immediate=is_critical,
                high_acuity=is_critical,
                esi_assigned=1 if is_critical else self._rng.randint(2, 3),
            )
            patients.append(patient)
        return patients

    def _compute_active_alarms(self, patients: List[Patient]) -> List[str]:
        alarms: List[str] = []
        for p in patients:
            if (
                p.vitals.heart_rate > 100
                or p.vitals.respiratory_rate > 20
                or p.vitals.spo2 < 92
            ):
                alarms.append(p.patient_id)
        return alarms

    def state(self) -> Observation:
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, object]]:
        self._state.clock += 1
        self.simulation_clock = self._state.clock
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

        # Clamp reward to strictly [0, 1] as per requirements
        reward_value = max(0.0, min(1.0, reward_value))

        info["preventable_deaths"] = preventable_deaths
        self.preventable_deaths += preventable_deaths
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
