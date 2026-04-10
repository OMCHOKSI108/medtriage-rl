---
title: MedTriage RL Simulation
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
- openenv
---

# Project MedTriage (OpenEnv)

This repository is an informational, non-clinical simulation concept for reinforcement learning research. It is not a medical device and provides no medical advice. For medical advice or diagnosis, consult a professional.

## Concept
MedTriage simulates an emergency department triage desk where an RL agent receives patient vitals, symptoms, and history, and must:
- Triage patients using the Emergency Severity Index (ESI) logic
- Allocate scarce resources (ICU beds, trauma bays, doctors)
- Prevent avoidable deterioration and deaths
- Handle sudden mass-casualty surges

The goal is high-impact multi-constraint optimization under uncertainty, with transparent, programmatic decision rules.

## ESI Logic (Programmatic)
The environment uses a strict, deterministic interpretation of ESI as a decision tree:

1. **ESI 1 (Immediate)**
   - Requires immediate life-saving intervention (e.g., cardiac arrest, severe trauma)
   - Must bypass queue and receive a resuscitation/trauma bed immediately

2. **ESI 2 (Emergency)**
   - High risk of deterioration
   - Includes a "vitals override" rule: if patient vitals cross defined bounds, they must be upgraded to ESI 2
   - Example override: heart rate > 100 bpm, respiratory rate > 20 / min, or SpO2 < 92%

3. **ESI 3-5 (Urgent to Minor)**
   - Determined by predicted resource consumption
   - More resources -> higher severity (lower ESI number)

## Observation, Action, and State
**Observation**
- `waiting_room`: list of masked patient summaries
- `bed_status`: counts of available ICU, trauma, and standard beds
- `active_alarms`: patient IDs that triggered critical vitals thresholds
- `simulation_clock`: integer time step

**Action**
- `triage_patient(patient_id, esi_level)`
- `allocate_bed(patient_id, bed_type)`
- `order_vitals_check(patient_id)`
- `no_op()`

**Internal State**
- Patient stability score decays if high-acuity patients wait too long
- Death event occurs when stability falls to 0
- Step advances `simulation_clock`

**OpenEnv API**
- `POST /reset` returns an initial observation
- `POST /step` advances the simulation and returns reward + done
- `GET /state` returns the current observation snapshot

## Reward and Scoring
Grader logic is task-specific. A typical score includes:
- Correct ESI assignments (accuracy)
- Penalties for preventable deaths
- Penalties for resource misuse
 - Partial progress for triage completion and bed allocation

Example scoring form:

$$
score = accuracy - 0.3 \cdot preventable\_deaths - 0.1 \cdot wasted\_resources
$$

Scores are clamped to $(0.01, 0.99)$ to avoid the Phase 2 out-of-range autograder issue.

The environment reward includes partial progress signals and is returned as a typed object:

$$
reward = 0.02 \cdot triaged + 0.03 \cdot bedded - 0.5 \cdot preventable\_deaths
$$

## Task Definitions (openenv.yaml)
See [openenv.yaml](openenv.yaml) for the required task metadata. Tasks included:
- Routine Resource Allocation
- Hidden Deterioration Triage
- Mass Casualty Surge

## Repository Layout
```
.
├── env_server.py
├── inference.py
├── openenv.yaml
├── requirements.txt
├── src
│   └── medtriage
│       ├── __init__.py
│       ├── models.py
│       └── sim.py
└── tasks
    ├── __init__.py
    ├── hidden_deterioration_triage
    │   ├── __init__.py
    │   └── grader.py
    ├── mass_casualty_surge
    │   ├── __init__.py
    │   └── grader.py
    └── routine_resource_allocation
        ├── __init__.py
        └── grader.py
```

## Run the Server
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the server:
    - `python -m uvicorn env_server:app --reload --port 7860`

## Test Inference Client
- `python inference.py`

The inference client reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from the environment
and emits structured logs in the `[START]`, `[STEP]`, `[END]` format.

It also supports `OPENAI_API_KEY` and `ENV_SERVER_URL` (default: http://127.0.0.1:7860) for local testing.

## OpenEnv Validation
Install `openenv-core`, then run:
- `openenv validate`
- `scripts/validate-submission.sh <your_space_url>`

## Environment Variables
See `.env.example` for the required variables used by the inference client.

## Baseline Scores
Baseline scores will vary with model and hardware. A deterministic heuristic policy is used
when model calls fail. Record your latest baseline scores here before submission:

- Routine Resource Allocation: TBD
- Hidden Deterioration Triage: TBD
- Mass Casualty Surge: TBD

## Hugging Face Spaces Deployment
1. Create a new Space (Docker) and tag it with `openenv`.
2. Push this repo to the Space.
3. Ensure the Space responds with HTTP 200 at `/reset` and `/state`.

## Notes for OpenEnv Phase 2
- All grader functions must return a float strictly within $(0.01, 0.99)$
- Each task must be defined in openenv.yaml with correct grader paths

## Next Steps
- Expand the simulation state machine in [src/medtriage/sim.py](src/medtriage/sim.py)
- Add scenario-specific generators per task
- Add unit tests for ESI and vitals override logic
# MedTriage Agent (Healthcare RL Environment)

MedTriage Agent is a reinforcement learning environment that simulates a hospital emergency room where an intelligent agent performs the role of a triage nurse. The system is designed to model real-world constraints such as limited ICU beds, limited medical staff, and unpredictable patient inflow. The agent receives patient information including vital signs, symptoms, and medical history, and must make decisions that directly impact patient outcomes.

The primary objective of the agent is to correctly prioritize patients based on severity. This involves identifying critical cases quickly and ensuring that high-risk patients receive immediate attention. At the same time, the agent must efficiently allocate scarce resources such as ICU beds and available doctors without causing preventable patient deterioration or death.

The environment introduces dynamic challenges such as sudden mass-casualty events, where multiple patients arrive at once. This tests the agent’s ability to adapt under pressure and make optimal decisions in highly constrained and time-sensitive situations. The agent must balance fairness, urgency, and resource limitations while maintaining overall system stability.

This project demonstrates the application of reinforcement learning in a high-impact domain. It combines decision-making under uncertainty, multi-constraint optimization, and real-time prioritization. The simulation provides a realistic and scalable framework for experimenting with intelligent healthcare systems.

Overall, MedTriage Agent highlights how AI can be used to support critical decision-making in emergency healthcare settings, making it a strong example of applied reinforcement learning in a socially meaningful context.
Overall, MedTriage Agent highlights how AI can be used to support critical decision-making in emergency healthcare settings, making it a strong example of applied reinforcement learning in a socially meaningful context.
Overall, MedTriage Agent highlights how AI can be used to support critical decision-making in emergency healthcare settings, making it a strong example of applied reinforcement learning in a socially meaningful context.
Overall, MedTriage Agent highlights how AI can be used to support critical decision-making in emergency healthcare settings, making it a strong example of applied reinforcement learning in a socially meaningful context.
