from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import TrafficAction, TrafficObservation

class TrafficEnv(EnvClient[TrafficAction, TrafficObservation, State]):
    """Client for the Traffic Signal RL Environment."""

    def _step_payload(self, action: TrafficAction) -> Dict:
        return {
            "action_type": action.action_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrafficObservation]:
        obs_data = payload.get("observation", {})
        observation = TrafficObservation(
            queue_lengths=obs_data.get("queue_lengths", {}),
            waiting_times=obs_data.get("waiting_times", {}),
            signal_phase=obs_data.get("signal_phase", "NS_GREEN"),
            time_since_last_change=obs_data.get("time_since_last_change", 0),
            emergency_presence=obs_data.get("emergency_presence", {}),
            pedestrian_requests=obs_data.get("pedestrian_requests", {}),
            task_difficulty=obs_data.get("task_difficulty", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
