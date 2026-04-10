from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    TRIAGE_PATIENT = "triage_patient"
    ALLOCATE_BED = "allocate_bed"
    ORDER_VITALS_CHECK = "order_vitals_check"
    NO_OP = "no_op"


class BedType(str, Enum):
    ICU = "icu"
    TRAUMA = "trauma"
    STANDARD = "standard"


class Vitals(BaseModel):
    heart_rate: int
    respiratory_rate: int
    spo2: int


class PatientSummary(BaseModel):
    patient_id: str
    age: int
    complaint: str
    vitals: Vitals
    resources_expected: int = Field(ge=0, le=5)
    esi_assigned: Optional[int] = Field(default=None, ge=1, le=5)


class BedStatus(BaseModel):
    icu: int
    trauma: int
    standard: int


class Observation(BaseModel):
    waiting_room: List[PatientSummary]
    bed_status: BedStatus
    active_alarms: List[str]
    simulation_clock: int


class Action(BaseModel):
    action_type: ActionType
    patient_id: Optional[str] = None
    esi_level: Optional[int] = Field(default=None, ge=1, le=5)
    bed_type: Optional[BedType] = None


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: "Reward"
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: Observation


class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
