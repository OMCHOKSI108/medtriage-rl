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
## Environment Variables (Competition)

For HF Space deployment and competition submission, the following environment variables are required:

| Variable | Description | Example |
|----------|-------------|---------|
| `API_BASE_URL` | LiteLLM proxy URL (injected by competition) | `https://litellm-proxy.com/v1` |
| `API_KEY` | API key for the proxy (injected by competition) | `sk-...` |
| `MODEL_NAME` | Model to use (optional, defaults to gpt-4o-mini) | `gpt-4o-mini` |
| `ENV_BASE_URL` | Environment server URL | `http://127.0.0.1:7860` |

**Important**: The competition injects `API_BASE_URL` and `API_KEY`. Do not hardcode your own OpenAI credentials - the submission will fail validation if it doesn't use the provided proxy.

## Repository Layout
```
.
├── env_server.py      # FastAPI server implementation
├── inference.py       # Baseline evaluation script
├── models.py          # Typed Pydantic models (Action, Observation, etc.)
├── client.py          # Standard OpenEnv client (EnvClient)
├── openenv.yaml       # Task metadata and grader definitions
├── Dockerfile         # Container definition (Port 7860)
├── requirements.txt   # Dependencies
├── src/medtriage/     # Simulation core logic
└── tasks/             # Procedural graders per task
```

## Baseline Reproducibility

Scores below are approximate. Run with seed=42 for deterministic episodes.

| Task | Difficulty | Max Steps | Baseline Score | Model |
|------|-----------|-----------|----------------|-------|
| routine_resource_allocation | easy | 12 | ~0.65 | gpt-4o-mini |
| hidden_deterioration_triage | medium | 12 | ~0.45 | gpt-4o-mini |
| mass_casualty_surge | hard | 15 | ~0.30 | gpt-4o-mini |

### Reproducing baseline

```bash
export API_BASE_URL=<injected-litellm-proxy-url>
export HF_TOKEN=<your-key>
export MODEL_NAME=gpt-4o-mini
python inference.py
```

## Deployment
1. **Containerized Execution**:
   - `docker build -t medtriage .`
   - `docker run -p 7860:7860 medtriage`
2. **Hugging Face Space**:
   - Tag your Space with `openenv`
   - For judged submissions, do not hardcode or manually override `API_BASE_URL` / `HF_TOKEN` with your own provider values in Space settings. The evaluator injects those at runtime.
   - Delete any stale `API_KEY` or `OPENAI_API_KEY` Space secrets so your run cannot silently use your own provider credentials instead of the evaluator proxy.

## OpenEnv Validation
The environment is fully compliant with the OpenEnv specification.
- `openenv validate` ➔ **PASSED**
- `scripts/validate-submission.sh` ➔ **PASSED**

---

# MedTriage Agent (Healthcare RL Environment)

MedTriage Agent is a reinforcement learning environment that simulates a hospital emergency room where an intelligent agent performs the role of a triage nurse. The system is designed to model real-world constraints such as limited ICU beds, limited medical staff, and unpredictable patient inflow. The agent receives patient information including vital signs, symptoms, and medical history, and must make decisions that directly impact patient outcomes.

The primary objective of the agent is to correctly prioritize patients based on severity. This involves identifying critical cases quickly and ensuring that high-risk patients receive immediate attention. At the same time, the agent must efficiently allocate scarce resources such as ICU beds and available doctors without causing preventable patient deterioration or death.

The environment introduces dynamic challenges such as sudden mass-casualty events, where multiple patients arrive at once. This tests the agent’s ability to adapt under pressure and make optimal decisions in highly constrained and time-sensitive situations. The agent must balance fairness, urgency, and resource limitations while maintaining overall system stability.

Overall, MedTriage Agent highlights how AI can be used to support critical decision-making in emergency healthcare settings, making it a strong example of applied reinforcement learning in a socially meaningful context.
