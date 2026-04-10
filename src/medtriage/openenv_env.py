from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from src.medtriage.models import Action, Observation, Reward, StepResponse
from src.medtriage.sim import MedTriageSim


@dataclass
class MedTriageOpenEnv:
    """Minimal OpenEnv-compatible wrapper around MedTriageSim."""

    sim: MedTriageSim

    @classmethod
    def create(cls) -> "MedTriageOpenEnv":
        return cls(sim=MedTriageSim())

    def reset(self) -> StepResponse:
        observation = self.sim.reset()
        reward = Reward(value=0.0, components={"reset": 0.0})
        return StepResponse(observation=observation, reward=reward, done=False, info={})

    def step(self, action: Action) -> StepResponse:
        observation, reward, done, info = self.sim.step(action)
        return StepResponse(observation=observation, reward=reward, done=done, info=info)

    def state(self) -> Observation:
        return self.sim.state()

    def close(self) -> None:
        return None
