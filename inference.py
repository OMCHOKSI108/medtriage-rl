"""Hackathon inference loop for the MedTriage OpenEnv environment.

Runs all 3 tasks (routine, deterioration, mass_casualty) sequentially using the OpenAI client.
Emits structured [START]/[STEP]/[END] logs per the hackathon spec.
"""

import os
import json
import time
import asyncio
from urllib import request as urllib_request
from openai import OpenAI

try:
    from client import MedTriageClient as MedTriageEnv
    from models import Action as MedTriageAction
    from models import ActionType
    _IMPORT_OK = True
    _IMPORT_ERROR = ""
except Exception as _import_err:
    _IMPORT_OK = False
    _IMPORT_ERROR = str(_import_err)

# ===========================================================================
# CONFIGURATION 
# ===========================================================================

# FIX 1: Safely format the injected API_BASE_URL for the OpenAI SDK
raw_url = os.environ["API_BASE_URL"].strip()
if not raw_url.startswith("http"):
    raw_url = f"http://{raw_url}"
if not raw_url.endswith("/v1") and not raw_url.endswith("/v1/"):
    raw_url = f"{raw_url.rstrip('/')}/v1"
API_BASE_URL = raw_url

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    API_KEY = os.environ.get("HF_TOKEN")
    if API_KEY:
        print("[WARN] API_KEY not set, falling back to HF_TOKEN", flush=True)

MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"
ENV_BASE_URL = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:7860").strip()

BENCHMARK_NAME          = "medtriage-er-simulator"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
REQUEST_MAX_RETRIES     = int(os.getenv("REQUEST_MAX_RETRIES", "2"))
MAX_RUNTIME_SECONDS     = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "1100"))
SUCCESS_SCORE_THRESHOLD = 0.6
TASK_MAX_REWARD = {
    "routine_resource_allocation": 0.30,
    "hidden_deterioration_triage": 0.45,
    "mass_casualty_surge": 0.75,
}

TASK_IDS = [
    "routine_resource_allocation",
    "hidden_deterioration_triage",
    "mass_casualty_surge",
]

TASK_MAX_STEPS = {
    "routine_resource_allocation": 12,
    "hidden_deterioration_triage": 12,
    "mass_casualty_surge":         15,
}

# ===========================================================================
# STRUCTURED LOGGING  (hackathon spec — do NOT change format)
# ===========================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ===========================================================================
# SERVER READINESS PROBE
# ===========================================================================

def wait_for_server(base_url: str, max_attempts: int = 30, delay: int = 4) -> bool:
    url = f"{base_url.rstrip('/')}/state"
    for _ in range(max_attempts):
        try:
            req = urllib_request.Request(
                url, method="GET", headers={"Accept": "application/json"}
            )
            with urllib_request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(delay)
    return False

# ===========================================================================
# PROMPT CONSTRUCTION
# ===========================================================================

SYSTEM_PROMPT = (
    "You are an elite ER triage agent managing patients and allocating beds.\n"
    "CRITICAL RULES:\n"
    "1. TRIAGE FIRST: If a patient has no ESI level yet, use 'triage_patient' "
    "   with esi_level (1=most critical, 5=minor).\n"
    "2. ALLOCATE BED: Once triaged, use 'allocate_bed' with bed_type "
    "   ('icu', 'trauma', or 'standard').\n"
    "3. OUTPUT ONLY JSON — no markdown, no explanation. "
    "   Required keys: action_type, patient_id. "
    "   Add esi_level (int) when triaging. Add bed_type (str) when allocating.\n"
    "Examples:\n"
    '  {"action_type": "triage_patient", "patient_id": "p1", "esi_level": 2}\n'
    '  {"action_type": "allocate_bed",   "patient_id": "p1", "bed_type": "icu"}\n'
)

def build_user_prompt(
    task_id: str,
    waiting_room: list[dict],
    bed_status: dict,
    active_alarms: list[str],
) -> str:
    room_lines = [
        f"id={p.get('patient_id')} age={p.get('age')} "
        f"complaint={p.get('complaint')} "
        f"HR={p.get('vitals', {}).get('heart_rate')} "
        f"SpO2={p.get('vitals', {}).get('spo2')} "
        f"esi={p.get('esi_assigned')}"
        for p in waiting_room
    ]
    room_block = " | ".join(room_lines) if room_lines else "nobody waiting"
    return (
        f"Task: {task_id}. "
        f"Waiting room: {room_block}. "
        f"Beds: ICU={bed_status.get('icu')} "
        f"Trauma={bed_status.get('trauma')} "
        f"Standard={bed_status.get('standard')}. "
        f"Active alarms: {active_alarms}."
    )

# ===========================================================================
# LLM CALL  
# ===========================================================================

def choose_action_with_llm(
    client: OpenAI,
    task_id: str,
    prompt: str,
) -> "MedTriageAction":
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
        stream=False,
    )

    raw_content = (completion.choices[0].message.content or "").strip()
    if not raw_content:
        return MedTriageAction(action_type=ActionType.NO_OP)

    try:
        if raw_content.startswith("```"):
            lines = [l for l in raw_content.split("\n") if not l.strip().startswith("```")]
            raw_content = "\n".join(lines)
        if "{" in raw_content and "}" in raw_content:
            start = raw_content.find("{")
            end = raw_content.rfind("}") + 1
            raw_content = raw_content[start:end]
        data = json.loads(raw_content)
        act_type = data.get("action_type", "no_op")
        action_payload = {"action_type": act_type}
        if "patient_id" in data:
            action_payload["patient_id"] = str(data["patient_id"])
        if act_type == "triage_patient" and "esi_level" in data:
            action_payload["esi_level"] = int(data["esi_level"])
        if act_type == "allocate_bed" and "bed_type" in data:
            action_payload["bed_type"] = str(data["bed_type"])
        return MedTriageAction(**action_payload)
    except Exception:
        return MedTriageAction(action_type=ActionType.NO_OP)

# ===========================================================================
# DETERMINISTIC FALLBACK 
# ===========================================================================

def choose_action_with_fallback(
    llm_action: "MedTriageAction | None",
    waiting_room: list[dict],
    bed_status: dict,
    triaged_patients: set[str],
) -> "MedTriageAction":
    # Accept a valid LLM action
    if (
        llm_action is not None
        and getattr(llm_action, "action_type", "no_op") != "no_op"
        and getattr(llm_action, "patient_id", None) is not None
    ):
        return llm_action

    if not waiting_room:
        return MedTriageAction(action_type=ActionType.NO_OP)

    # Triage the first un-triaged patient, severity based on vitals
    for p in waiting_room:
        pid = p.get("patient_id")
        if pid not in triaged_patients and p.get("esi_assigned") is None:
            vitals = p.get("vitals", {})
            hr   = vitals.get("heart_rate", 80)
            spo2 = vitals.get("spo2", 98)
            if (hr and hr > 130) or (spo2 and spo2 < 88):
                esi = 1
            elif (hr and hr > 100) or (spo2 and spo2 < 92):
                esi = 2
            else:
                esi = 3
            return MedTriageAction(
                action_type=ActionType.TRIAGE_PATIENT,
                patient_id=pid,
                esi_level=esi,
            )

    # Allocate beds for already-triaged patients
    for p in waiting_room:
        pid = p.get("patient_id")
        if p.get("esi_assigned") is not None:
            esi = p.get("esi_assigned", 3)
            if esi <= 1 and bed_status.get("icu", 0) > 0:
                bed_type = "icu"
            elif esi <= 2 and bed_status.get("trauma", 0) > 0:
                bed_type = "trauma"
            else:
                bed_type = "standard"
            return MedTriageAction(
                action_type=ActionType.ALLOCATE_BED,
                patient_id=pid,
                bed_type=bed_type,
            )

    return MedTriageAction(action_type=ActionType.NO_OP)

# ===========================================================================
# SINGLE-TASK RUNNER
# ===========================================================================

async def run_task(
    llm_client: OpenAI,
    env: "MedTriageEnv",
    task_id: str,
    start_time: float,
) -> None:
    max_steps = TASK_MAX_STEPS.get(task_id, 12)
    task_name = f"medtriage-{task_id}"
    rewards:  list[float] = []
    steps_taken = 0
    score       = 0.01
    success     = False
    triaged_patients: set[str] = set()

    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        try:
            obs = env.reset(task_id=task_id)
        except Exception as exc:
            log_step(step=1, action="reset()", reward=0.0, done=True, error=str(exc))
            return

        for step in range(1, max_steps + 1):
            if time.time() - start_time >= MAX_RUNTIME_SECONDS:
                log_step(
                    step=step, action="timeout_guard",
                    reward=0.0, done=True, error="runtime limit reached",
                )
                break

            obs_dict     = obs.model_dump() if hasattr(obs, "model_dump") else obs
            waiting_room = obs_dict.get("waiting_room", [])

            if not waiting_room:
                break

            bed_status_dict = obs_dict.get("bed_status", {})
            prompt     = build_user_prompt(
                task_id=task_id,
                waiting_room=waiting_room,
                bed_status=bed_status_dict,
                active_alarms=obs_dict.get("active_alarms", []),
            )

            try:
                llm_action = choose_action_with_llm(llm_client, task_id, prompt)
            except Exception as llm_exc:
                # FIX 2: Crash loudly instead of hiding the error behind a fallback.
                # If there's an issue with the proxy, we WANT to see this stack trace!
                print(f"[FATAL ERROR] LLM call failed at step {step}: {llm_exc}", flush=True)
                raise 

            action = choose_action_with_fallback(
                llm_action=llm_action,
                waiting_room=waiting_room,
                bed_status=bed_status_dict,
                triaged_patients=triaged_patients,
            )

            if (
                getattr(action, "action_type", None) == "triage_patient"
                and getattr(action, "patient_id", None)
            ):
                triaged_patients.add(action.patient_id)

            try:
                obs, reward_obj, done, info = env.step(action)
            except Exception as exc:
                log_step(step=step, action="env.step()", reward=0.0, done=True, error=str(exc))
                break

            reward_val  = float(getattr(reward_obj, "value", reward_obj) or 0.0)
            rewards.append(reward_val)
            steps_taken = step

            action_type = getattr(action, "action_type", "unknown")
            pid         = getattr(action, "patient_id", None)
            esi         = getattr(action, "esi_level",  None)
            bed         = getattr(action, "bed_type",   None)

            log_step(
                step=step,
                action=f"{action_type}(patient_id={pid},esi={esi},bed={bed})",
                reward=reward_val,
                done=bool(done),
                error=None,
            )

            if done:
                break

        max_reward = TASK_MAX_REWARD.get(task_id, 0.50)
        raw_score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(raw_score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ===========================================================================
# MAIN
# ===========================================================================

async def main() -> None:
    start_time = time.time()

    if not _IMPORT_OK:
        for task_id in TASK_IDS:
            log_start(task=f"medtriage-{task_id}", env=BENCHMARK_NAME, model=MODEL_NAME)
            log_step(1, "import", 0.0, True, error=_IMPORT_ERROR)
            log_end(False, 1, 0.01, [0.01])
        return

    missing_vars = []
    # API_BASE_URL check handled at the top of the file
    if not API_KEY:
        missing_vars.append("API_KEY")

    if missing_vars:
        err = f"Missing required env vars: {', '.join(missing_vars)}"
        print(f"[ERROR] {err}", flush=True)
        for task_id in TASK_IDS:
            log_start(task=f"medtriage-{task_id}", env=BENCHMARK_NAME, model=MODEL_NAME)
            log_step(1, "init", 0.0, True, error=err)
            log_end(False, 1, 0.01, [0.0])
        return

    print(f"[INFO] API_BASE_URL = {API_BASE_URL!r}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME!r}",   flush=True)
    print(f"[INFO] ENV_BASE_URL = {ENV_BASE_URL!r}", flush=True)

    print("[INFO] Waiting for env server...", flush=True)
    if not wait_for_server(ENV_BASE_URL):
        err = f"Env server at {ENV_BASE_URL} never became ready"
        print(f"[ERROR] {err}", flush=True)
        for task_id in TASK_IDS:
            log_start(task=f"medtriage-{task_id}", env=BENCHMARK_NAME, model=MODEL_NAME)
            log_step(1, "wait_for_server", 0.0, True, error=err)
            log_end(False, 1, 0.01, [0.0])
        return
    print("[INFO] Env server is ready.", flush=True)

    try:
        llm_client = OpenAI(
            base_url=API_BASE_URL, 
            api_key=API_KEY,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=REQUEST_MAX_RETRIES,
        )
    except Exception as e:
        err = f"Failed to create OpenAI client: {e}"
        print(f"[ERROR] {err}", flush=True)
        for task_id in TASK_IDS:
            log_start(task=f"medtriage-{task_id}", env=BENCHMARK_NAME, model=MODEL_NAME)
            log_step(1, "init_client", 0.0, True, error=err)
            log_end(False, 1, 0.01, [0.0])
        return

    try:
        env = MedTriageEnv(base_url=ENV_BASE_URL)
    except Exception as e:
        err = f"Failed to create env client: {e}"
        print(f"[ERROR] {err}", flush=True)
        for task_id in TASK_IDS:
            log_start(task=f"medtriage-{task_id}", env=BENCHMARK_NAME, model=MODEL_NAME)
            log_step(1, "init_env", 0.0, True, error=err)
            log_end(False, 1, 0.01, [0.0])
        return

    for task_id in TASK_IDS:
        if time.time() - start_time >= MAX_RUNTIME_SECONDS:
            break
        await run_task(llm_client, env, task_id, start_time)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[ERROR] Unhandled inference failure: {exc}", flush=True)
        raise