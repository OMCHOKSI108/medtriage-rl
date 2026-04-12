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

MedTriage is an emergency-room triage environment built for OpenEnv reinforcement learning evaluation. It simulates constrained clinical operations where agents must prioritize patients and allocate limited beds under time pressure.

This project is strictly for simulation and research benchmarking. It is not a clinical decision system.

## Environment Description and Motivation

Emergency triage is a high-stakes scheduling problem with competing objectives:
- Escalate high-acuity patients quickly.
- Avoid preventable deterioration while patients wait.
- Allocate ICU, trauma, and standard beds efficiently.

MedTriage models those trade-offs through deterministic patient dynamics, task-specific graders, and structured step-level feedback. The benchmark is intended to stress decision quality, consistency, and prioritization discipline.

## Observation Space

Each step returns an Observation object with:

| Field | Type | Description |
|---|---|---|
| waiting_room | list[PatientSummary] | Current patients waiting for triage/bed allocation |
| bed_status | BedStatus | Available bed counts: icu, trauma, standard |
| active_alarms | list[str] | Patient IDs that violate vitals thresholds |
| simulation_clock | int | Current simulation timestep |

PatientSummary fields:
- patient_id: str
- age: int
- complaint: str
- vitals: {heart_rate, respiratory_rate, spo2}
- resources_expected: int (0-5)
- esi_assigned: int or null (1-5)

## Action Space

The agent can emit the following actions:

| Action | Required Fields | Description |
|---|---|---|
| triage_patient | patient_id, esi_level | Assign ESI level (1-5) |
| allocate_bed | patient_id, bed_type | Allocate icu, trauma, or standard bed |
| order_vitals_check | patient_id | Fetch latest vitals for a patient |
| no_op | none | No-op step |

Validation constraints come from Pydantic models:
- esi_level must be 1-5
- bed_type must be one of icu, trauma, standard

## Tasks and Expected Difficulty

Task metadata is defined in openenv.yaml.

| Task ID | Name | Difficulty | Expected Behavior |
|---|---|---|---|
| routine_resource_allocation | Routine Resource Allocation | easy | Correctly triage routine low-acuity flow and avoid wasteful allocations |
| hidden_deterioration_triage | Hidden Deterioration Triage | medium | Identify a deceptive case requiring urgent escalation from vitals cues |
| mass_casualty_surge | Mass Casualty Surge | hard | Prioritize critical patients during surge with tight ICU/trauma capacity |

## Reward and Grading Overview

Simulator step reward components:
- +0.02 * triaged patients
- +0.03 * patients assigned to beds
- -0.5 * preventable deaths

Reward is clamped to [0.0, 1.0].

Task graders compute final score differently per task (each clamped to [0.01, 0.99]):
- routine_resource_allocation: accuracy minus death penalty
- hidden_deterioration_triage: hidden case escalation plus partial credit
- mass_casualty_surge: ICU placement quality plus survival component

## Setup

### Prerequisites
- Python 3.10+
- Docker (for container execution / HF Space)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables used by inference.py

| Variable | Required | Default in code | Notes |
|---|---|---|---|
| API_BASE_URL | no | https://router.huggingface.co/v1 | OpenAI-compatible endpoint |
| LLM_API_BASE_URL | no | API_BASE_URL | Optional override for LLM endpoint |
| MODEL_NAME | no | meta-llama/Llama-3.1-8B-Instruct | Default baseline model |
| HF_TOKEN | yes | none | Required API key for model calls |
| ENV_BASE_URL | no | http://127.0.0.1:7860 | Local env server URL |
| ENV_SERVER_URL | no | fallback for ENV_BASE_URL | Compatibility alias |

## Usage

### 1) Run environment server

```bash
python -m uvicorn env_server:app --host 0.0.0.0 --port 7860
```

### 2) Run baseline inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=<your-hf-token>
export ENV_BASE_URL=http://127.0.0.1:7860
python inference.py
```

Expected log line families:
- [START] task=... env=... model=...
- [STEP] step=... action=... reward=... done=... error=...
- [END] success=... steps=... score=... rewards=...

## Baseline Scores (Reference)

Reference baseline profile for the default setup (Meta Llama 3.1 8B Instruct, seed=42, local environment):

| Task | Difficulty | Max Steps | Reference Score |
|---|---|---:|---:|
| routine_resource_allocation | easy | 12 | ~0.65 |
| hidden_deterioration_triage | medium | 12 | ~0.45 |
| mass_casualty_surge | hard | 15 | ~0.30 |

These are practical reference values, not fixed constants; exact values may vary by provider latency and generation behavior.

## API Endpoints

The FastAPI server exposes:
- POST /reset
- POST /step
- GET /state

Default app port: 7860

## Project Structure

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
│   ├── openenv_env.py
│   └── sim.py
└── tasks/
	├── routine_resource_allocation/grader.py
	├── hidden_deterioration_triage/grader.py
	└── mass_casualty_surge/grader.py
```

## Docker

```bash
docker build -t medtriage .
docker run -p 7860:7860 medtriage
```
