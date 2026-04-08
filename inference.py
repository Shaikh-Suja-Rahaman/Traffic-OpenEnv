import os
import textwrap
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

from client import TrafficEnv
from models import TrafficAction

load_dotenv()

# Required Env Variables
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")


def observation_to_dict(observation: object) -> dict:
    if isinstance(observation, dict):
        return observation
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    return dict(getattr(observation, "__dict__", {}))


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))

def build_user_prompt(step: int, last_action: str, last_reward: float, history: List[str], obs: dict) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last action taken: {last_action!r}
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        
        Current State:
        Task Difficulty: {obs.get('task_difficulty')}
        Signal Phase: {obs.get('signal_phase')}
        Time Since Last Change: {obs.get('time_since_last_change')}
        Queue Lengths: {obs.get('queue_lengths')}
        Waiting Times: {obs.get('waiting_times')}
        Emergency Presence: {obs.get('emergency_presence')}
        Pedestrian Requests: {obs.get('pedestrian_requests')}
        
        Decide your next action carefully to optimize the traffic. Your action must simply be a raw string of either 'KEEP_PHASE' or 'SWITCH_PHASE'.
        """
    ).strip()

def run_agent():
    # If variables are missing in local dev, fallback for dry-run
    api_key = HF_TOKEN or "dummy"
    llm_base_url = API_BASE_URL or "https://api.openai.com/v1"
    model = MODEL_NAME or "gpt-4o"
    env_url = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000")
    llm_timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "45"))
    episode_step_guard = int(os.environ.get("EPISODE_STEP_GUARD", "200"))
    
    client = OpenAI(api_key=api_key, base_url=llm_base_url, timeout=llm_timeout_s)
    env = TrafficEnv(base_url=env_url).sync()

    with env:
        for episode in range(3):
            # Start
            try:
                result = env.reset()
                obs_data = observation_to_dict(result.observation)
            except Exception as e:
                print(f"Failed to reset environment: {e}")
                break

            task = obs_data.get("task_difficulty", f"task_{episode}")
            env_name = "traffic_rl"

            print(f"[START] task={task} env={env_name} model={model}", flush=True)

            step = 0
            done = False
            rewards = []
            raw_rewards = []
            history = []
            last_action = "None"
            last_reward = 0.0
            success = False
            score = 0.0

            while not done and step < episode_step_guard:
                prompt = build_user_prompt(step, last_action, last_reward, history, obs_data)

                try:
                    # LLM call
                    messages = [
                        {"role": "system", "content": "You are a highly efficient traffic signal controller. Output only 'KEEP_PHASE' or 'SWITCH_PHASE'."},
                        {"role": "user", "content": prompt},
                    ]
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=20,
                        timeout=llm_timeout_s,
                    )
                    action = completion.choices[0].message.content.strip().upper()

                    action = "SWITCH_PHASE" if "SWITCH" in action else "KEEP_PHASE"

                    error_val = "null"
                except Exception as e:
                    action = "KEEP_PHASE"
                    error_val = str(e).replace('\n', ' ')

                step += 1
                last_action = action

                try:
                    # Step the environment through a persistent OpenEnv session.
                    result = env.step(TrafficAction(action_type=action))
                    obs_data = observation_to_dict(result.observation)

                    raw_reward = float(result.reward if result.reward is not None else obs_data.get("reward", 0.0))
                    reward = clamp_unit_interval(raw_reward)
                    done_val = bool(result.done if result.done is not None else obs_data.get("done", False))
                except Exception as e:
                    raw_reward = 0.0
                    reward = 0.0
                    done_val = True
                    error_val = str(e).replace('\n', ' ')

                raw_rewards.append(raw_reward)
                rewards.append(reward)
                last_reward = reward
                done = done_val

                # Formatting as required by the spec exactly
                done_str = str(done_val).lower()
                print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_val}", flush=True)
                history.append(f"Phase={obs_data.get('signal_phase')} -> Act:{action} -> Rwd:{reward:.2f}")

            if not done and step >= episode_step_guard:
                done = True
                print(
                    f"[STEP] step={step} action=KEEP_PHASE reward=0.00 done=true error=Episode step guard reached ({episode_step_guard})",
                    flush=True,
                )

            # Compute Score inside [0.0, 1.0]. Easy tasks map higher positive reward. Hard tasks map lower.
            meta = obs_data.get("metadata", {})
            cumulative = meta.get("cumulative_reward", sum(raw_rewards))
            # Hard cap for grading
            score = max(0.0, min(1.0, (cumulative + 500) / 1000.0))
            success = score >= 0.3

            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(f"[END] success={str(success).lower()} steps={step} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_agent()
