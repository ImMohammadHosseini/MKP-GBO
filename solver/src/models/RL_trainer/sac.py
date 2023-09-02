
"""

"""
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
import numpy as np
from ..mlp import Critic, Value

class SAC():
    def __init__ (
        self,
        actor_model: torch.nn.Module,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        scale: int = 2,
    ):
        self.actor_model = actor_model
        self.critic_model1 = Critic(self.actor_model.rowNum, self.actor_model.colNum,
                                    device=self.actor_model.device, name='critic_model1')
        self.critic_model2 = Critic(self.actor_model.rowNum, self.actor_model.colNum,
                                    device=self.actor_model.device, name='critic_model2')
        self.value_model1 = Value(self.actor_model.rowNum, self.actor_model.colNum,
                                  device=self.actor_model.device, name='value_model1')
        self.value_model2 = Value(self.actor_model.rowNum, self.actor_model.colNum,
                                  device=self.actor_model.device, name='value_model2')
        self.actor_optimizer = Adam(self.actor_model.parameters(), 
                                      lr=0.001)
        self.critic_optimizer1 = Adam(self.critic_model1.parameters(), 
                                      lr=0.001)
        self.critic_optimizer2 = Adam(self.critic_model2.parameters(), 
                                      lr=0.001)
        self.value_optimizer1 = Adam(self.value_model1.parameters(), 
                                      lr=0.001)
        self.value_optimizer2 = Adam(self.value_model2.parameters(), 
                                      lr=0.001)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.scale = scale
        #Replay buffer
        self.buffer_size= int(1e6)
        self._transitions_stored = 0
        
        self.memory_observation = np.zeros((self.buffer_size,16,6))
        self.memory_action = np.zeros(self.buffer_size)
        self.memory_reward = np.zeros(self.buffer_size)
        self.memory_newObservation = np.zeros((self.buffer_size,16,6))
        self.memory_done = np.zeros(self.buffer_size, dtype=np.bool)
        self.weights = np.zeros(self.buffer_size)
        
    def save_step ( 
        self, 
        observation: torch.tensor, 
        action: int, 
        reward: int,
        new_observation: torch.tensor, 
        done: bool, 
    ):
        index = self._transitions_stored % self.buffer_size
        self.memory_observation[index] = observation
        self.memory_newObservation[index] = new_observation
        self.memory_action[index] = action
        self.memory_reward[index] = reward
        self.memory_done[index] = done

        self._transitions_stored += 1
    
    def sample_buffer(
        self,
    ):
        max_stored = min(self._transitions_stored, self.buffer_size)

        batch = np.random.choice(max_stored, self.batch_size)

        observations = self.memory_observation[batch]
        newObservations = self.memory_newObservation[batch]
        actions = self.memory_action[batch]
        rewards = self.memory_reward[batch]
        dones = self.memory_done[batch]

        return observations, actions, rewards, newObservations, dones
    
    def step_act (
        self,
        observation: torch.tensor, 
    ):
        action, _ = self.actor_model.sample_normal(observation.unsqueeze(0))
        return action.squeeze(0)
    
    def train (
        self,
    ):
        #torch.autograd.set_detect_anomaly(True)

        if self._transitions_stored < self.batch_size:
            return

        observations, mactions, rewards, new_observations, dones = \
                self.sample_buffer()

        observations = torch.tensor(observations, dtype=torch.float).to(self.actor_model.device)
        mactions = torch.tensor(mactions, dtype=torch.float).to(self.actor_model.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor_model.device)
        new_observations = torch.tensor(new_observations, dtype=torch.float).to(self.actor_model.device)
        dones = torch.tensor(dones).to(self.actor_model.device)
        
        values = self.value_model1(observations).view(-1)
        new_values = self.value_model2(new_observations).view(-1)
        new_values[dones] = 0.0

        actions, log_probs = self.actor_model.sample_normal(observations, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_model1.forward(observations, actions)
        q2_new_policy = self.critic_model2.forward(observations, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value_optimizer1.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * mse_loss(values, value_target)
        value_loss.backward(retain_graph=True)
        self.value_optimizer2.step()

        actions, log_probs = self.actor_model.sample_normal(observations, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_model1.forward(observations, actions)
        q2_new_policy = self.critic_model2.forward(observations, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        
        q_hat = self.scale*rewards + self.gamma*new_values
        q1_old_policy = self.critic_model1.forward(observations, mactions.unsqueeze(1)).view(-1)
        q2_old_policy = self.critic_model2.forward(observations, mactions.unsqueeze(1)).view(-1)
        critic_1_loss = 0.5 * mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * mse_loss(q2_old_policy, q_hat)

        critic_loss = (critic_1_loss + critic_2_loss).detach()
        critic_loss.requires_grad=True
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        #print(critic_loss)
        critic_loss.backward()
        #print('f')

        self.critic_optimizer1.step()
        self.critic_optimizer2.step()
        #print('ff')

        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        value_params2 = self.value_model2.named_parameters()
        value_params1 = self.value_model1.named_parameters()

        target_value_state_dict = dict(value_params2)
        value_state_dict = dict(value_params1)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.value_model2.load_state_dict(value_state_dict)
    
    