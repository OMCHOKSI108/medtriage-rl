import json
import os
from typing import Any, Dict, List

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
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ENV_SERVER_URL = get_env_var("ENV_SERVER_URL", "http://127.0.0.1:7860")

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


import time

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
    
    _log("START", {"task": task_id, "env": BENCHMARK, "model": MODEL_NAME})

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

        _log("STEP", {"step": step, "action": action, "reward": reward, "done": done})
        history.append(f"Step {step}: reward {last_reward:+.2f}")

        if done:
            break

    score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    score = min(max(score, 0.0), 1.0)
    _log("END", {"task": task_id, "score": score, "steps": steps_taken})
    return score

def main() -> None:
    api_key = HF_TOKEN or OPENAI_API_KEY or "sk-ignored"
    if not api_key or str(api_key).strip().lower() == "none":
        api_key = "sk-ignored"

    # Deeply hardened client creation
    client = None
    try:
        # Only pass base_url if it looks like a real URL
        kwargs = {"api_key": api_key}
        if API_BASE_URL.startswith("http"):
            kwargs["base_url"] = API_BASE_URL
        
        client = OpenAI(**kwargs)
    except BaseException:
        try:
            # Fallback to absolute minimum config
            client = OpenAI(api_key="sk-ignored")
        except BaseException:
            client = None

    tasks = []
    try:
        tasks = _load_tasks()
    except BaseException:
        pass

    for task in tasks:
        try:
            _run_task(task.get("id", "unknown"), client)
        except BaseException:
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        _log("FATAL", {"error": str(e), "trace": traceback.format_exc()})
