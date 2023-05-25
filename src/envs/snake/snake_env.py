from __future__ import annotations

from collections import deque
from typing import Any, Tuple

import gym
import numpy as np
from gym import spaces

from src.envs.snake.snake_game import SnakeGame


class SnakeEnv(gym.Env):
    def __init__(self, kill_reward_weight: float = 0.9, num_stacked_images: int = 4, render_mode: str | None = None) -> None:
        self._kill_reward_weight = kill_reward_weight
        self._num_stacked_images = num_stacked_images
        self.render_mode = render_mode
        
        # instantiate the game
        self._snake_game = SnakeGame()
        if self.render_mode == "human":
            self._snake_game.set_mode("play")
        else:
            self._snake_game.set_mode("training")
        
        # set obs space
        self._image_buffer = deque(maxlen=self._num_stacked_images)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(64 * self._num_stacked_images, 64, 3))
        # set action space
        self.action_space = spaces.Discrete(5)
        
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        self._snake_game.start()
        self._snake_game.update()
        self._clear_image_buffer()
        for _ in range(self._num_stacked_images): # type: ignore
            self._add_current_image()
        obs = self._get_stacked_image()
        self._prev_food_score = 0
        self._prev_kill_score = 0
        return obs, dict()
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        # set action
        action = action.item()
        if action == 0:
            self._snake_game.set_move("none")
        elif action == 1:
            self._snake_game.set_move("up")
        elif action == 2:
            self._snake_game.set_move("down")
        elif action == 3:
            self._snake_game.set_move("left")
        elif action == 4:
            self._snake_game.set_move("right")
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # update the game
        self._snake_game.update()
        
        # get the observation
        self._add_current_image()
        next_obs = self._get_stacked_image()
        
        # set terminated
        # terminated = self._snake_game.is_dead
        terminated = self._snake_game.gameover()
        
        # set reward
        _, food_score, kill_score = self._snake_game.get_score()
        if terminated:
            reward = -1.0
        else:
            reward = ((1.0 - self._kill_reward_weight) * (food_score - self._prev_food_score) + \
                self._kill_reward_weight * (kill_score - self._prev_kill_score)) * 0.99
            reward += 0.01
            self._prev_food_score = food_score
            self._prev_kill_score = kill_score
            
        return next_obs, reward, terminated, False, dict()
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._snake_game.screen_image
        else:
            raise NotImplementedError
    
    def _add_current_image(self):
        self._image_buffer.append(self._snake_game.screen_image.astype(np.float32) / 255.0)
        
    def _clear_image_buffer(self):
        self._image_buffer.clear()
    
    def _get_stacked_image(self) -> np.ndarray:
        return np.concatenate(self._image_buffer, axis=0)
 