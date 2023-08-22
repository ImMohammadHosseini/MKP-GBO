
"""

"""
import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from src.data_structure.state_prepare import StatePrepare
from copy import deepcopy

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
    ):        
        super().__init__()
        
        
    def setStatePrepare (
        self,
        stateDataClasses: StatePrepare,
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
        
        self.observation = torch.tensor(observation, dtype=torch.float32,
                                   device=self.device)
        info = self._get_info()

        return observation, info
    
    def convertToOneHot(self, dat, fixed_part, ksNum):
        alloc = []
        for i in dat:
            oneHot = [0] * (ksNum+1); alist = i.tolist()[-(ksNum+1):]
            oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
        new_dat_oneHot = torch.cat((fixed_part, torch.FloatTensor(alloc)), dim=1)
        return new_dat_oneHot
    
    def step (self, action):
        new_matrix = deepcopy(self.observation)
        new_matrix.requires_grad = True
        optimizer = torch.optim.Adam([new_matrix] , lr=0.8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        iteration = 0; equal = 0
        fixed_part = deepcopy(new_matrix.data[:,0:-(self.statePrepare.knapsackObsSize+1)])
        alloc_old = deepcopy(new_matrix.data[:,-(self.statePrepare.knapsackObsSize+1):])
        optimizer.zero_grad(); action.backward(); optimizer.step(); scheduler.step()
        new_matrix.data = self.convertToOneHot(new_matrix.data, fixed_part, self.statePrepare.knapsackObsSize+1)
        new_matrix.requires_grad = False 
        decision = []
        for i, row in enumerate(new_matrix): 
            decision_part = row[-(self.statePrepare.knapsackObsSize+1):].tolist()
            if i == self.statePrepare.instanceObsSize:
                break
            if decision_part[-1] == 1:
                continue
            decision.append((i, decision_part.index(max(decision_part))))
        
        reward = self.reward(decision)
        self.statePrepare.updateMatrix(new_matrix)
        
        info = self._get_info()
        
        self.observation = self._get_obs()
        
        self.observation = torch.tensor(self.observation, dtype=torch.float32)
        
        
        
        return self.observation, reward, terminated, info
    
    
    
    def reward (self, decisions):
        newScore = self.statePrepare.getScore(decisions)
        reward = newScore - self.score
        self.score = newScore
        return reward
    

    
    
    
    