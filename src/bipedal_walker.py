import torch
import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.agent as agent
from aine_drl.factory import (AgentFactory, AINEInferenceFactory,
                              AINETrainFactory)
from aine_drl.policy import GaussianPolicy

LEARNING_RATE = 3e-4

class BipedalWalkerPPONet(nn.Module, agent.PPOSharedNetwork):
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        self.hidden_feature = 256
        
        self.encoding_layer = nn.Sequential(
            nn.Linear(obs_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_feature)
        )
        
        self.actor = GaussianPolicy(self.hidden_feature, num_actions)
        self.critic = nn.Linear(self.hidden_feature, 1)
    
    def model(self) -> nn.Module:
        return self
    
    def forward(self, obs: aine_drl.Observation) -> tuple[aine_drl.PolicyDist, torch.Tensor]:
        encoding = self.encoding_layer(obs.items[0])
        policy_dist = self.actor(encoding)
        state_value = self.critic(encoding)
        return policy_dist, state_value
        
class BipedalWalkerPPOFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.PPOConfig(**config_dict)
        
        network = BipedalWalkerPPONet(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.continuous
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=LEARNING_RATE
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
                
        return agent.PPO(config, network, trainer, env.num_envs)
                
def train():
    ppo_config_path = "config/bipedal_walker_v3_ppo.yaml"
    AINETrainFactory.from_yaml(ppo_config_path) \
        .make_env() \
        .make_agent(BipedalWalkerPPOFactory()) \
        .ready() \
        .train() \
        .close()
        
def inference():
    ppo_config_path = "config/bipedal_walker_v3_ppo.yaml"
    AINEInferenceFactory.from_yaml(ppo_config_path) \
        .make_env() \
        .make_agent(BipedalWalkerPPOFactory()) \
        .ready() \
        .inference() \
        .close()