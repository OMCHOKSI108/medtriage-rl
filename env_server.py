from __future__ import annotations

from fastapi import FastAPI

from src.medtriage.models import Reward, StateResponse, StepRequest, StepResponse
from src.medtriage.sim import MedTriageSim

app = FastAPI(title="MedTriage ER Simulator")

_sim = MedTriageSim()


@app.post("/reset", response_model=StepResponse)
def reset() -> StepResponse:
    observation = _sim.reset()
    return StepResponse(
        observation=observation,
        reward=Reward(value=0.0, components={"reset": 0.0}),
        done=False,
        info={"note": "reset"},
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    observation, reward, done, info = _sim.step(request.action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(state=_sim.state())
