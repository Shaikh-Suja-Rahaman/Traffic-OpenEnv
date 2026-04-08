from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict, Literal

class TrafficAction(Action):
    """Action for the Traffic Signal RL environment."""
    action_type: Literal["KEEP_PHASE", "SWITCH_PHASE"] = Field(..., description="Action to take.")

class TrafficObservation(Observation):
    """Observation from the Traffic Signal RL environment."""
    queue_lengths: Dict[str, int] = Field(default_factory=dict, description="Queue lengths for N, S, E, W lanes.")
    waiting_times: Dict[str, float] = Field(default_factory=dict, description="Total waiting times for N, S, E, W lanes.")
    signal_phase: str = Field(default="NS_GREEN", description="Current green phase ('NS_GREEN' or 'EW_GREEN').")
    time_since_last_change: int = Field(default=0, description="Time steps since the last phase change.")
    emergency_presence: Dict[str, bool] = Field(default_factory=dict, description="Emergency vehicle presence per lane.")
    pedestrian_requests: Dict[str, bool] = Field(default_factory=dict, description="Pedestrian crossing requests.")
    task_difficulty: str = Field(default="easy", description="Current task difficulty.")
