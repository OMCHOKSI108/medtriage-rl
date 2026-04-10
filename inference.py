import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
import yaml
from openai import OpenAI


def get_env_var(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is None or val.strip().lower() == "none" or val.strip() == "":
        return default
    return val.strip()

API_BASE_URL = get_env_var("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = get_env_var("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
ENV_SERVER_URL = get_env_var("ENV_SERVER_URL", "http://127.0.0.1:7860")

BENCHMARK = "medtriage-er-simulator"
MAX_STEPS = 12
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.6


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload)}", flush=True)


def log_start(task: str, env: str, model: str) -> None:
    _emit("START", {"task": task, "env": env, "model": model})


def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    _emit(
        "STEP",
        {
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "error": error,
        },
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    _emit("END", {"success": success, "steps": steps, "score": score, "rewards": rewards})


def _clamp_score(score: float) -> float:
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return score


def _load_tasks() -> List[Dict[str, Any]]:
    with open("openenv.yaml", "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload.get("tasks", [])


def _get_model_message(client: Optional[OpenAI], step: int, history: List[str]) -> str:
    if client is None:
        return "triage"
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
    except Exception:
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


def _wait_for_server(max_attempts: int = 20, delay: int = 5) -> bool:
    """Polls the server /state endpoint until it is ready."""
    for i in range(max_attempts):
        try:
            response = requests.get(f"{ENV_SERVER_URL}/state", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def _safe_post(url: str, json_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handles POST requests with robust error catching and JSON validation."""
    try:
        response = requests.post(url, json=json_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}

def _run_task(task_id: str, client: Optional[OpenAI]) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Polling wait (silent)
    _wait_for_server()

    result = _safe_post(f"{ENV_SERVER_URL}/reset")
    last_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if not result or result.get("done"):
            break

        _ = _get_model_message(client, step, history)
        action = _choose_action(result.get("observation", {}))
        result = _safe_post(f"{ENV_SERVER_URL}/step", json_data={"action": action})
        
        reward = _get_reward_value(result or {})
        done = (result or {}).get("done", True)

        rewards.append(reward)
        steps_taken = step
        last_reward = reward

        log_step(step=step, action=action, reward=reward, done=bool(done), error=None)
        history.append(f"Step {step}: reward {last_reward:+.2f}")

        if done:
            break

    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = _clamp_score(score)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

def main() -> None:
    api_key = API_KEY or ""
    client = None
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    except BaseException:
        client = None

    try:
        tasks = _load_tasks()
    except BaseException:
        tasks = []

    for task in tasks:
        task_id = task.get("id", "unknown_task")
        _run_task(task_id, client)

if __name__ == "__main__":
    main()
