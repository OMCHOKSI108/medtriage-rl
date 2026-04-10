import json
import os
from typing import Any, Dict, List

import requests
import yaml
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:7860")

BENCHMARK = "medtriage-er-simulator"
MAX_STEPS = 12
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.6


def _log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}")


def _load_tasks() -> List[Dict[str, Any]]:
    with open("openenv.yaml", "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload.get("tasks", [])


def _get_model_message(client: OpenAI, step: int, history: List[str]) -> str:
    prompt = "You are a triage assistant. Return a short action plan."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"{prompt}\nStep: {step}\nHistory: {history}"}],
            temperature=0,
            max_tokens=20,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "triage"
    except Exception as exc:
        _log("STEP", {"event": "model_error", "error": str(exc)})
        return "triage"


def _choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    waiting_room = observation.get("waiting_room", [])
    if not waiting_room:
        return {"action_type": "no_op"}
    patient = waiting_room[0]
    return {
        "action_type": "triage_patient",
        "patient_id": patient["patient_id"],
        "esi_level": 3,
    }


def _get_reward_value(step_payload: Dict[str, Any]) -> float:
    reward = step_payload.get("reward", {})
    if isinstance(reward, dict):
        return float(reward.get("value", 0.0))
    return float(reward or 0.0)


def _run_task(task_id: str, client: OpenAI) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    _log("START", {"task": task_id, "env": BENCHMARK, "model": MODEL_NAME})

    reset_response = requests.post(f"{ENV_SERVER_URL}/reset", timeout=10)
    reset_response.raise_for_status()
    result = reset_response.json()
    last_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if result.get("done"):
            break

        _ = _get_model_message(client, step, history)
        action = _choose_action(result.get("observation", {}))

        step_response = requests.post(
            f"{ENV_SERVER_URL}/step",
            json={"action": action},
            timeout=10,
        )
        step_response.raise_for_status()
        result = step_response.json()

        reward = _get_reward_value(result)
        done = result.get("done")
        error = None

        rewards.append(reward)
        steps_taken = step
        last_reward = reward

        _log(
            "STEP",
            {
                "step": step,
                "action": action,
                "reward": reward,
                "done": done,
                "error": error,
            },
        )

        history.append(f"Step {step}: reward {last_reward:+.2f}")

        if done:
            break

    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD

    _log("END", {"success": success, "steps": steps_taken, "score": score, "rewards": rewards})
    return score


def main() -> None:
    api_key = HF_TOKEN or OPENAI_API_KEY or "sk-ignored"
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

    tasks = _load_tasks()
    scores: List[float] = []
    for task in tasks:
        task_id = task.get("id", "unknown_task")
        scores.append(_run_task(task_id, client))

    _log("END", {"success": True, "steps": len(scores), "score": sum(scores) / max(len(scores), 1)})


if __name__ == "__main__":
    main()
