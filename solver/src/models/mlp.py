
"""

"""
import torch
import torch.nn as nn
from typing import List, Optional
from torch.distributions.normal import Normal



class actor_MLP(nn.Module):
    def __init__(self, 
        rowNum: int, 
        colNum: int,
        hidden_dims: List = [254, 128, 128, 64],
        device = torch.device('cpu'),
        name = 'actor_MLP'
    ):
        super(actor_MLP, self).__init__()
        self.name = name
        self.device = device
        self.rowNum = rowNum
        self.colNum =colNum
        self.reparam_noise = 1e-6
        
        modules = []
        input_dim = rowNum * colNum
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Softplus())
            )
            input_dim = h_dim
        self.mlp = nn.Sequential(*modules).to(self.device)
        
        self.mu = nn.Linear(input_dim, 1, device=self.device)
        self.sigma = nn.Linear(input_dim, 1, device=self.device)
        

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = self.mlp(x)
        
        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma
        
    
    def sample_normal(self, observation, reparameterize=True):
        mu, sigma = self.forward(observation)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.sigmoid(actions).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs= log_probs.unsqueeze(-1)
        #print(log_probs.size())
        #print(log_probs)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
class Value(nn.Module):
    def __init__(self, 
        rowNum: int, 
        colNum: int,
        hidden_dims: List = [254, 128, 128, 64],
        device = torch.device('cpu'),
        name = 'value_model'
    ):
        super(Value, self).__init__()
        self.name = name
        self.device = device
        
        modules = []
        input_dim = rowNum * colNum
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Softplus())
            )
            input_dim = h_dim
        self.mlp = nn.Sequential(*modules).to(self.device)
        
        self.outer = nn.Linear(input_dim, 1, device=self.device)
        
    def forward(self, observation):
        observation = observation.flatten(start_dim=1)
        observation = self.mlp(observation)
        observation = self.outer(observation)
        return observation
    
class Critic(nn.Module):
    def __init__(self, 
        rowNum: int, 
        colNum: int,
        hidden_dims: List = [254, 128, 128, 64],
        device = torch.device('cpu'),
        name = 'critic_model'
    ):
        super(Critic, self).__init__()
        self.name = name
        self.device = device
        
        modules = []
        input_dim = rowNum * colNum + 1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Softplus())
            )
            input_dim = h_dim
        self.mlp = nn.Sequential(*modules).to(self.device)
        self.outer = nn.Linear(input_dim, 1, device=self.device)
        
    def forward(self, observation, action):
        observation = observation.flatten(start_dim=1)
        observation = torch.cat([observation, action], dim=1)
        observation = self.mlp(observation)
        observation = self.outer(observation)
        return observation