from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from openenv.core.env_client import EnvClient
from models import Action, Observation, Reward, StepRequest, StepResponse


class MedTriageClient(EnvClient):
    """
    Client for the MedTriage environment.
    Connects to the FastAPI server and handles typed observations/actions.
    """

    def reset(self, **kwargs: Any) -> Observation:
        response = self._session.post(f"{self.base_url}/reset", json=kwargs or {})
        response.raise_for_status()
        data = StepResponse.model_validate(response.json())
        return data.observation

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        request = StepRequest(action=action)
        response = self._session.post(
            f"{self.base_url}/step",
            json=request.model_dump()
        )
        response.raise_for_status()
        data = StepResponse.model_validate(response.json())
        return data.observation, data.reward, data.done, data.info

    def get_state(self) -> Observation:
        response = self._session.get(f"{self.base_url}/state")
        response.raise_for_status()
        return Observation.model_validate(response.json()["state"])


# Export for OpenEnv validation
Env = MedTriageClient
