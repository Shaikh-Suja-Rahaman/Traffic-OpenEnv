import uuid
import logging
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import TrafficAction, TrafficObservation
from simulator import TrafficSimulator

logger = logging.getLogger(__name__)

REWARD_MIN_EXCLUSIVE = 0.01
REWARD_MAX_EXCLUSIVE = 0.99

class TrafficEnvironment(Environment):
    """
    Traffic Signal RL Environment.
    The agent manages traffic light phases to optimize throughput and fairness.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    MAX_STEPS = 20
    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.current_task_idx = 0
        self._init_simulator()
        
    def _init_simulator(self):
        diff = self.DIFFICULTIES[min(self.current_task_idx, len(self.DIFFICULTIES) - 1)]
        self.simulator = TrafficSimulator(difficulty=diff)
        self.cumulative_reward = 0.0

    def _normalize_step_reward(self, raw_reward: float) -> float:
        # Map raw simulator rewards to the grading-friendly [0,1] band first,
        # then enforce strict exclusivity via epsilon margins.
        mapped = (raw_reward + 500.0) / 1000.0
        return max(REWARD_MIN_EXCLUSIVE, min(REWARD_MAX_EXCLUSIVE, mapped))

    def reset(self) -> TrafficObservation:
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._init_simulator()
        return self._get_observation(reward=0.0, done=False)

    def _get_observation(self, reward: float, done: bool) -> TrafficObservation:
        return TrafficObservation(
            queue_lengths=self.simulator.queues.copy(),
            waiting_times=self.simulator.wait_times.copy(),
            signal_phase=self.simulator.signal_phase,
            time_since_last_change=self.simulator.time_since_last_change,
            emergency_presence=self.simulator.emergency.copy(),
            pedestrian_requests=self.simulator.pedestrians.copy(),
            task_difficulty=self.simulator.difficulty,
            reward=reward,
            done=done,
            metadata={
                "cumulative_reward": self.cumulative_reward,
                "step": self._state.step_count
            }
        )

    def step(self, action: TrafficAction) -> TrafficObservation:
        # Check if we already hit done
        if self._state.step_count >= self.MAX_STEPS:
            # Reached end of current task, reset for the next task automatically or just halt?
            # Standard RL environments return done. Up to client to call reset().
            return self._get_observation(reward=REWARD_MIN_EXCLUSIVE, done=True)
            
        raw_step_reward = self.simulator.step(action.action_type)
        step_reward = self._normalize_step_reward(raw_step_reward)
        self.cumulative_reward += raw_step_reward
        self._state.step_count += 1
        
        done = (self._state.step_count >= self.MAX_STEPS)
        
        if done:
            # At the end of the episode, map raw cumulative reward to a 0.0 - 1.0 score.
            # Empirical mapping (since we don't know the exact max bounds without running an agent, 
            # we'll frame it so anything >= 0 is mapped roughly into a normalized scale)
            # Easy task gets mostly positive rewards. Medium gets mixed. Hard gets heavily penalized.
            score = max(REWARD_MIN_EXCLUSIVE, min(REWARD_MAX_EXCLUSIVE, (self.cumulative_reward + 500) / 1000.0))
            # The hackathon asks for graders to produce score between 0-1.
            # We overwrite the final step reward with this score for the client to read, 
            # or the client computes based on cumulative reward. 
            # To provide dense signal, step_reward is returned. The log script logs the 'score' independently.
            
            # Auto-advance task index so next reset gives next difficulty
            if self.current_task_idx < len(self.DIFFICULTIES) - 1:
                self.current_task_idx += 1
            
        return self._get_observation(reward=step_reward, done=done)

    @property
    def state(self) -> State:
        return self._state
