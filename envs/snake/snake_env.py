from __future__ import annotations
from typing import Optional, Tuple, Any
import gym
from gym import spaces

class SnakeEnv(gym.Env):
    def __init__(self) -> None:
        # TODO: instantiate the snake game
        
        # set obs space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1000, 1000, 3))
        # set action space
        self.action_space = spaces.Discrete(5)
        
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        raise NotImplementedError
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        raise NotImplementedError