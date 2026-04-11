import json
import os
import time
from typing import Any, Dict, List

import requests
import yaml
from openai import OpenAI

# =========================
# ENV CONFIG
# =========================
_raw_base = os.getenv("API_BASE_URL", "").strip()
if not _raw_base:
    API_BASE_URL = "https://api.openai.com/v1"
elif not _raw_base.startswith("http"):
    API_BASE_URL = f"http://{_raw_base}"
else:
    API_BASE_URL = _raw_base

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"
# Support both token names, typical of Hugging Face Spaces and standard endpoints.
API_KEY = os.getenv("HF_TOKEN", "").strip() or os.getenv("API_KEY", "").strip()

ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")
BENCHMARK = "medtriage-er-simulator"

MAX_STEPS = 12
SUCCESS_SCORE_THRESHOLD = 0.6
MAX_TOTAL_REWARD = 1.0  # Used for normalization

# =========================
# STRICT LOGGING FORMAT
# =========================
def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# =========================
# HELPERS
# =========================
def wait_for_server():
    for _ in range(20):
        try:
            r = requests.get(f"{ENV_SERVER_URL}/state", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def safe_post(url: str, payload: Dict[str, Any] | None = None):
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[DEBUG] POST error: {e}", flush=True)
        return {}

def load_tasks():
    try:
        with open("openenv.yaml", "r") as f:
            return yaml.safe_load(f).get("tasks", [])
    except Exception:
        # Fallback to the first task if yaml is missing/corrupt
        return [{"id": "routine_resource_allocation"}]

# =========================
# LLM DECISION MAKER
# =========================
def get_llm_action(client: OpenAI, observation: Dict, history: List[str]) -> Dict:
    prompt = f"""
You are an ER triage expert. Use the following observation to pick the next action.

Observation:
{json.dumps(observation, indent=2)}

History:
{history[-3:] if history else 'None'}

Choose the most critical patient and assign ESI level (1-5).
Return ONLY a valid JSON object.

Example:
{{
  "patient_id": "P-5",
  "esi_level": 1
}}
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )

        text = (completion.choices[0].message.content or "").strip()
        
        # Clean potential markdown formatting
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        action_data = json.loads(text.strip())
        patient_id = action_data.get("patient_id")
        esi_level = action_data.get("esi_level", 3)
        
        # Ensure primitive types
        if not patient_id:
            raise ValueError("No patient_id provided from LLM")
            
        return {
            "action_type": "triage_patient",
            "patient_id": str(patient_id),
            "esi_level": int(esi_level),
        }

    except Exception as e:
        print(f"[DEBUG] LLM parsing/request failed: {e}", flush=True)
        # Safe fallback action to prevent crashing
        waiting = observation.get("waiting_room", [])
        if waiting:
            return {
                "action_type": "triage_patient",
                "patient_id": waiting[0]["patient_id"],
                "esi_level": 3,
            }
        return {"action_type": "no_op"}

def get_reward(result: Dict) -> float:
    reward = result.get("reward", {})
    if isinstance(reward, dict):
        return float(reward.get("value", 0.0))
    return float(reward or 0.0)

# =========================
# MAIN TASK LOOP
# =========================
def run_task(task_id: str, client: OpenAI):
    history = []
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task_id)
    wait_for_server()

    result = safe_post(f"{ENV_SERVER_URL}/reset")

    try:
        for step in range(1, MAX_STEPS + 1):
            if not result or result.get("done"):
                break

            observation = result.get("observation", {})
            action = get_llm_action(client, observation, history)
            
            result = safe_post(f"{ENV_SERVER_URL}/step", payload={"action": action})

            reward = get_reward(result)
            done = result.get("done", True)

            rewards.append(reward)
            steps_taken = step
            
            # Format action for logging exactly as sample
            if action.get("action_type") == "triage_patient":
                action_str = f"triage('{action.get('patient_id')}', {action.get('esi_level')})"
            else:
                action_str = "no_op()"
                
            log_step(step, action_str, reward, done, None)

            history.append(f"step {step}: reward {reward:+.2f}")
            if done:
                break

        # Calculate final score normalized to [0, 1]
        total_reward = sum(rewards)
        score = min(max(total_reward / MAX_TOTAL_REWARD, 0.0), 1.0) if MAX_TOTAL_REWARD > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during run_task: {e}", flush=True)
    finally:
        # Mandatory: Always emit END block, even if an exception breaks the loop
        log_end(success, steps_taken, score, rewards)

def main():
    if not API_KEY:
        print("[FATAL] Missing API_KEY or HF_TOKEN environment variables.", flush=True)
        return

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[FATAL] Could not initialize OpenAI client: {e}", flush=True)
        return

    # Warmup call
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "ping"}],
            max_tokens=5,
        )
    except Exception as e:
         print(f"[DEBUG] proxy warmup failed: {e}", flush=True)

    tasks = load_tasks()
    for task in tasks:
        task_id = task.get("id", "unknown_task")
        try:
            run_task(task_id, client)
        except Exception as e:
            print(f"[ERROR] Task {task_id} externally failed: {e}", flush=True)

if __name__ == "__main__":
    main()