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

# MedTriage ER Simulator (OpenEnv)

MedTriage is a reinforcement learning environment that models emergency-room triage under constrained resources.

The environment is designed for research and benchmarking, not clinical use. It evaluates whether an agent can make timely, consistent decisions when patient risk, bed capacity, and arrival pressure change over time.

## Environment Motivation

Real triage settings require balancing two competing goals:
- Clinical urgency: critically ill patients must be identified and escalated quickly.
- Resource stewardship: ICU and trauma capacity is limited and cannot be over-allocated.

MedTriage captures this tension through deterministic simulation rules and task-specific graders. The benchmark focuses on policy quality under pressure, especially when deterioration signals are subtle or sudden surges occur.

## Observation Space

Each environment step returns an observation with the following fields:

| Field | Type | Description |
|---|---|---|
| waiting_room | list[PatientSummary] | Queue of current patients and their vitals/metadata |
| bed_status | object | Available beds by type: icu, trauma, standard |
| active_alarms | list[str] | Patient IDs currently triggering alarm conditions |
| simulation_clock | int | Current simulation timestep |

PatientSummary includes:
- patient_id
- age
- complaint
- vitals (heart_rate, respiratory_rate, spo2)
- resources_expected
- esi_assigned (nullable)

## Action Space

Supported actions:

| Action | Required Fields | Description |
|---|---|---|
| triage_patient | patient_id, esi_level | Assign ESI level (1-5) to a patient |
| allocate_bed | patient_id, bed_type | Assign patient to icu, trauma, or standard bed |
| order_vitals_check | patient_id | Request a vitals refresh for a patient |
| no_op | none | No state-changing action |

Notes:
- ESI levels are validated in the action model (1-5).
- Bed type is constrained to icu, trauma, standard.

## Tasks and Difficulty

Tasks are defined in openenv.yaml and graded independently:

| Task ID | Name | Difficulty | Objective |
|---|---|---|---|
| routine_resource_allocation | Routine Resource Allocation | easy | Triage routine flow and avoid unnecessary resource overuse |
| hidden_deterioration_triage | Hidden Deterioration Triage | medium | Detect a patient that appears mild but requires escalation based on vitals |
| mass_casualty_surge | Mass Casualty Surge | hard | Handle a high-volume arrival wave with strict critical-care prioritization |

## Setup

### 1) Requirements
- Python 3.10+
- Docker (for container run/HF Space)

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Required environment variables for inference.py

| Variable | Required | Default | Purpose |
|---|---|---|---|
| API_BASE_URL | yes | https://api.openai.com/v1 | OpenAI-compatible endpoint |
| MODEL_NAME | yes | gpt-4o-mini | Model identifier used for generation |
| HF_TOKEN | yes | none | API token used as OpenAI api_key |
| ENV_BASE_URL | no | http://127.0.0.1:7860 | Environment server URL |

HF_TOKEN is mandatory. inference.py exits early if it is missing.

## Usage

### Run environment server

```bash
python -m uvicorn env_server:app --host 0.0.0.0 --port 7860
```

### Run baseline inference

```bash
export API_BASE_URL=<your-endpoint>
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=<your-token>
export ENV_BASE_URL=http://127.0.0.1:7860
python inference.py
```

The script emits structured logs in this format:
- [START] task=... env=... model=...
- [STEP] step=... action=... reward=... done=... error=...
- [END] success=... steps=... score=... rewards=...

## Baseline Scores

Approximate baseline (gpt-4o-mini, default step limits):

| Task | Difficulty | Max Steps | Baseline Score |
|---|---|---:|---:|
| routine_resource_allocation | easy | 12 | ~0.65 |
| hidden_deterioration_triage | medium | 12 | ~0.45 |
| mass_casualty_surge | hard | 15 | ~0.30 |

Scores are expected to vary by provider/model latency and generation behavior.

## Repository Layout

```text
.
├── inference.py
├── env_server.py
├── client.py
├── models.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── src/medtriage/
└── tasks/
```

## Docker

Build and run:

```bash
docker build -t medtriage .
docker run -p 7860:7860 medtriage
```

## Validation

- openenv validate
- scripts/validate-submission.sh

Both checks should pass before submission.
