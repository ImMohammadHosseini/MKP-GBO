
"""

"""
import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
        no_change_times: int = 10,
    ):        
        super().__init__()
        self.no_change_times = no_change_times
        
        
    def setStatePrepare (
        self,
        stateDataClasses,
    ):
        self.statePrepare = stateDataClasses
        
    def _get_obs (self) -> dict:
        allocateMatrix = self.statePrepare.getObservation()
        return allocateMatrix
    
    def _get_info (self):
        return ""
         
    def reset (self): 
        self.score = 0
        self.no_change = 0
        self.statePrepare.reset()
        observation = self._get_obs()
        
        observation = torch.tensor(observation, dtype=torch.float32)#self.
        info = self._get_info()

        return observation, info#self.
    
    def convertToOneHot(self, dat, fixed_part, ksNum):
        alloc = []
        for i in dat:
            oneHot = [0] * (ksNum+1); alist = i.tolist()[-(ksNum+1):]
            oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
        new_dat_oneHot = torch.cat((fixed_part, torch.FloatTensor(alloc)), dim=1)
        return new_dat_oneHot
    
    def step (self, observation, action, model):
        #print('1',observation)
        optimizer = torch.optim.Adam([observation] , lr=0.8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        fixed_part = deepcopy(observation.data[:,0:-(self.statePrepare.knapsackObsSize+1)])#self.
        alloc_old = deepcopy(observation.data[:,-(self.statePrepare.knapsackObsSize+1):])#self.
        #a, _ = model.forward(observation)#self.
        #print('r', observation.requires_grad )#self.
        #print(observation.grad)
        #print(observation.is_leaf)

        optimizer.zero_grad(); action.backward(); optimizer.step(); scheduler.step()
        #print('grad',observation.grad)
        #print(observation)
        observation.data = self.convertToOneHot(observation.data, fixed_part, self.statePrepare.knapsackObsSize)
        #print('2',observation)
        
        decision = []
        for inst_id, row in enumerate(observation): 
            decision_part = row[-(self.statePrepare.knapsackObsSize+1):].cpu().detach().numpy()
            if inst_id == self.statePrepare.instanceObsSize:
                break
            if decision_part[-1] == 1 or (decision_part==0.0).all():
                continue
            ks_id = np.where(decision_part == 1)[0][0]
            decision.append((inst_id, ks_id))
        reward = self.reward(decision)
        if reward == 0:
            self.no_change += 1
        self.statePrepare.updateMatrix(observation)
        
        info = self._get_info()
        
        self.observation = self._get_obs()
        
        self.observation = torch.tensor(self.observation, dtype=torch.float32)
        
        if self.no_change == self.no_change_times: terminated = True
        else: terminated = False
        
        return self.observation, reward, terminated, info
    
    def reward (self, decisions):
        newScore = self.statePrepare.getScore(decisions)
        
        reward = newScore - self.score
        self.score = newScore
        return reward
    

    
    
    
    