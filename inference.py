import os
import time
import requests
import textwrap
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Required Env Variables
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.environ.get("HF_TOKEN")

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
    base_url = API_BASE_URL or "https://api.openai.com/v1"
    model = MODEL_NAME or "gpt-4o"
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    env_url = "https://sujarahaman-traffic-rl.hf.space"
    
    for episode in range(3):
        # Start
        try:
            res = requests.post(f"{env_url}/reset")
            res.raise_for_status()
            result = res.json()
            obs_data = result if "task_difficulty" in result else result.get("observation", {})
        except Exception as e:
            print(f"Failed to reset environment: {e}")
            break
            
        task = obs_data.get("task_difficulty", f"task_{episode}")
        env_name = "traffic_rl"
        
        print(f"[START] task={task} env={env_name} model={model}", flush=True)
        
        step = 0
        done = False
        rewards = []
        history = []
        last_action = "None"
        last_reward = 0.0
        success = False
        score = 0.0
        
        while not done:
            prompt = build_user_prompt(step, last_action, last_reward, history, obs_data)
            
            try:
                # LLM call
                messages = [
                    {"role": "system", "content": "You are a highly efficient traffic signal controller. Output only 'KEEP_PHASE' or 'SWITCH_PHASE'."},
                    {"role": "user", "content": prompt},
                ]
                completion = client.chat.completions.create(model=model, messages=messages, max_tokens=20)
                action = completion.choices[0].message.content.strip().upper()
                
                if "SWITCH" in action:
                    action = "SWITCH_PHASE"
                else:
                    action = "KEEP_PHASE"
                    
                error_val = "null"
            except Exception as e:
                action = "KEEP_PHASE"
                error_val = str(e).replace('\n', ' ')
            
            step += 1
            last_action = action
            
            try:
                # Step the environment
                res = requests.post(f"{env_url}/step", json={"action": {"action_type": action}})
                res.raise_for_status()
                result = res.json()
                
                obs_data = result if "task_difficulty" in result else result.get("observation", {})
                
                reward = float(obs_data.get("reward", result.get("reward", 0.0)))
                done_val = bool(obs_data.get("done", result.get("done", False)))
            except Exception as e:
                reward = 0.0
                done_val = True
                error_val = str(e).replace('\n', ' ')
                
            rewards.append(reward)
            last_reward = reward
            done = done_val
            
            # Formatting as required by the spec exactly
            done_str = str(done_val).lower()
            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_val}", flush=True)
            history.append(f"Phase={obs_data.get('signal_phase')} -> Act:{action} -> Rwd:{reward:.2f}")

        # Compute Score inside [0.0, 1.0]. Easy tasks map higher positive reward. Hard tasks map lower. 
        meta = obs_data.get("metadata", {})
        cumulative = meta.get("cumulative_reward", sum(rewards))
        # Hard cap for grading
        score = max(0.0, min(1.0, (cumulative + 500) / 1000.0))
        success = score >= 0.3 
        
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_agent()
