import math

import aine_drl
import torch
import torch.nn as nn
import torch.optim as optim
from aine_drl import Observation, PolicyDist
from aine_drl.agent import PPO, Agent, PPOConfig, PPOSharedNetwork
from aine_drl.agent.agent import Agent
from aine_drl.env import Env, GymEnv
from aine_drl.factory import AgentFactory, AINETrainFactory
from aine_drl.policy import CategoricalPolicy
from gym.vector import AsyncVectorEnv

from envs.snake import SnakeEnv


class SnakePPONet(nn.Module, PPOSharedNetwork):
    def __init__(self, img_obs_shape: tuple, num_actions: int) -> None:
        super().__init__()
        
        height = img_obs_shape[0]
        width = img_obs_shape[1]
        channel = img_obs_shape[2]
        
        conv_out_shape = self.conv_output_shape((height, width), kernel_size=8, stride=4)
        conv_out_shape = self.conv_output_shape(conv_out_shape, kernel_size=4, stride=2)
        conv_out_shape = self.conv_output_shape(conv_out_shape, kernel_size=3, stride=1)
        conv_flattened_features = conv_out_shape[0] * conv_out_shape[1] * 64
        
        self.encoding_layer = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_flattened_features, 512),
            nn.ReLU(),
        )
        
        self.actor = CategoricalPolicy(512, num_actions)
        self.critic = nn.Linear(512, 1)
    
    def model(self) -> nn.Module:
        return self
        
    def forward(self, obs: Observation) -> tuple[PolicyDist, torch.Tensor]:
        image_obs = obs.items[0].permute(0, 3, 1, 2)
        encoding = self.encoding_layer(image_obs)
        policy_dist = self.actor(encoding)
        state_value = self.critic(encoding)
        return policy_dist, state_value
    
    @staticmethod
    def conv_output_shape(
        h_w: tuple[int, int],
        kernel_size: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
    ):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = math.floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        w = math.floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
        )
        return h, w
    
class SnakePPOMaker(AgentFactory):
    def make(self, env: Env, config_dict: dict) -> Agent:
        config = PPOConfig(**config_dict)
        
        network = SnakePPONet(
            img_obs_shape=env.obs_spaces[0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=3e-4
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        return PPO(config, network, trainer, env.num_envs)
    
def train():
    aine_factory = AINETrainFactory.from_yaml("config/snake_ppo.yaml")
    
    num_envs = aine_factory.num_envs
    seed = aine_factory.seed
    env = GymEnv(AsyncVectorEnv([
        lambda: SnakeEnv() for _ in range(num_envs)
    ]), seed=seed)
    
    aine_factory.set_env(env) \
        .make_agent(SnakePPOMaker()) \
        .ready() \
        .train() \
        .close()
