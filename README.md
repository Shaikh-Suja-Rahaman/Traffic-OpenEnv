---
title: Traffic Signal RL Setting
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
app_port: 8000
---

# OpenEnv Traffic Signal RL 

A complete, real-world compliant Reinforcement Learning OpenEnv for a Traffic Signal optimization task.

## Environment Details
This environment challenges agents to balance throughput, queues, and queue-starvation across a 4-way intersection (North/South vs East/West). Harder difficulties introduce sporadic deterministic pedestrian behavior and emergency vehicle overrides that heavily penalize stalled traffic flows.

## State Space (Observation)
The environment provides dense dict metrics per step:
- `queue_lengths`: Waiting cars per lane (N, S, E, W)
- `waiting_times`: Accumulated wait-ticks of queues (preventing starvation)
- `signal_phase`: The currently active flow (e.g. `NS_GREEN`, `EW_GREEN`)
- `time_since_last_change`: Time steps elapsed on current green
- `emergency_presence`: Per-lane flags for emergency response
- `pedestrian_requests`: Active crosswalk hazards restricting flow
- `task_difficulty`: Curricular difficulty (easy/medium/hard)

## Action Space
- `action_type`: A discrete logic gate strictly returning `KEEP_PHASE` or `SWITCH_PHASE`.

## Setup
### Local Testing
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app
```

### Baseline Inference Validation
The inference runner now uses a stateful OpenEnv client session (WebSocket), which is the recommended OpenEnv interaction pattern for multi-step episodes.

Set the following variables before running:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-..."
export OPENENV_BASE_URL="http://127.0.0.1:8000"
# optional safety knobs
# export LLM_TIMEOUT_S="45"
# export EPISODE_STEP_GUARD="200"
python inference.py
```

For Hugging Face Spaces deployments, point `OPENENV_BASE_URL` to your Space URL, for example:
```bash
export OPENENV_BASE_URL="https://<your-space>.hf.space"
python inference.py
```

*Note: A dummy API token will fallback gracefully but emit "error" states as expected in logs.*

### Validation
Passes Hugging Face Spaces deployment & OpenEnv structural parameters:
```bash
python -m openenv validate
```
# Traffic-OpenEnv
