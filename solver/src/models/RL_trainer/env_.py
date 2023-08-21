
"""

"""
import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from src.data_structure.state_prepare import StatePrepare

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
        dim: int,
        info: dict,
        knapsackNum: int,
        instanceNum: int,
        device = "cpu",
    ):
        self.dim = dim
        '''self.observation_space = spaces.Dict(
            {
                "knapsack": spaces.Box(low=np.array(
                    [[[info['CAP_LOW'], info['CAP_LOW'], info['CAP_LOW'], 
                     info['CAP_LOW'], info['CAP_LOW'],0]]*knapsackObsSize]*main_batch_size), 
                                       high=np.array(
                    [[[info['CAP_HIGH'], info['CAP_HIGH'], info['CAP_HIGH'],
                     info['CAP_HIGH'], info['CAP_HIGH'],0]]*knapsackObsSize]*main_batch_size), 
                                       shape=(main_batch_size, knapsackObsSize,
                                              self.dim), dtype=int),
                "instance_value": spaces.Box(low=np.array(
                    [[[info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['VALUE_LOW']]]*instanceObsSize]*main_batch_size), 
                                       high=np.array(
                    [[[info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['WEIGHT_HIGH'],
                     info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], 
                     info['VALUE_HIGH']]]*instanceObsSize]*main_batch_size), 
                                       shape=(main_batch_size, instanceObsSize,
                                              self.dim), dtype=int),
            }
        )
        self.action_space = spaces.Box(low=np.array([[0,0]] * \
                                                    knapsackObsSize), 
                                       high=np.array([[knapsackObsSize, 
                                                       instanceObsSize]] * \
                                                     knapsackObsSize), 
                                       dtype=int
        )'''
        
        super().__init__()
        self.device = device
        self.info = info
        self.knapsackNum = knapsackNum
        self.instanceNum = instanceNum
        
    def setStatePrepare (
        self,
        stateDataClasses: StatePrepare,
    ):
        self.statePrepares = stateDataClasses
        
    def _get_obs (self) -> dict:
        '''batchCaps = np.zeros((0,30,4)); 
        batchWeightValues = np.zeros((0,30,4))
        for statePrepare in self.statePrepares:
            stateCaps, stateWeightValues = statePrepare.getObservation()
            batchCaps = np.append(batchCaps, np.expand_dims(stateCaps, 0), 0) 
            batchWeightValues = np.append(batchWeightValues, 
                                          np.expand_dims(stateWeightValues, 0), 0)'''
        stateCaps, stateWeightValues = self.statePrepares[0].getObservation()
        batchCaps = np.expand_dims(stateCaps, 0)
        batchWeightValues = np.expand_dims(stateWeightValues, 0)
        return {"knapsack": batchCaps, "instance_value":batchWeightValues}
        
    def _get_info (self):
        return ""
        
        
    def reset (self): 
        self.no_change = 0
        for statePrepare in self.statePrepares: statePrepare.reset()
        externalObservation = self._get_obs()
        SOD = np.array([1.]*4)
        EOD = np.array([[[2.]*4]]*self.main_batch_size)

        shape = externalObservation["instance_value"].shape
        sod_instance_value = np.zeros((shape[0], shape[1]+1, shape[2]))
        for index in range(self.main_batch_size):
           sod_instance_value[index] = np.insert(externalObservation[
               "instance_value"][index], self.statePrepares[index].pad_len, SOD, axis=0)
        
        externalObservation = np.append(np.append(sod_instance_value, EOD, axis=1), 
                                            np.append(externalObservation["knapsack"], 
                                             EOD, axis=1),axis=1)
        #externalObservation = np.append(externalObservation["instance_value"], 
        #                                externalObservation["knapsack"], axis=1)
        info = self._get_info()

        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device)#.unsqueeze(dim=0)
        #print(externalObservation.size())
        return externalObservation, info
    
    def step (self, step_actions):
        externalRewards = []
        terminated = False
        for index in range(self.main_batch_size):
            invalid_action_end_index = max(np.where(step_actions[index] == -1)[0], default=-1)
            #print('max ', invalid_action_end_index)
            if invalid_action_end_index == step_actions.shape[1]-1: self.no_change += 1
            else: self.no_change=0
            externalRewards.append(self.statePrepares[index].changeNextState(
                step_actions[index][invalid_action_end_index+1:]))               
            terminated = terminated or self.statePrepares[index].is_terminated()
        
        #terminated = terminated or self.no_change >= self.no_change_long
        
        info = self._get_info()

        #if terminated:
        #    return None, externalRewards, terminated, info
        
        externalObservation = self._get_obs()
        SOD = np.array([1.]*self.dim)
        EOD = np.array([[[2.]*self.dim]]*self.main_batch_size)

        shape = externalObservation["instance_value"].shape
        sod_instance_value = np.zeros((shape[0], shape[1]+1, shape[2]))
        for index in range(self.main_batch_size):
           sod_instance_value[index] = np.insert(externalObservation[
               "instance_value"][index], self.statePrepares[index].pad_len, SOD, axis=0)

        
        externalObservation = np.append(np.append(sod_instance_value, EOD, axis=1),
                                            np.append(externalObservation["knapsack"], 
                                                      EOD, axis=1),axis=1)
        #externalObservation = np.append(externalObservation["instance_value"], 
        #                                externalObservation["knapsack"], axis=1)
        #print(externalObservation)
        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device)
        #print(externalObservation.size())

        
        return externalObservation, externalRewards, terminated, info
    
    
    def response_decode (self, responce):
        pass
    
    def final_score (self):
        scores = []; remain_cap_ratios = []
        for statePrepare in self.statePrepares:
            score = 0
            remain_cap_ratio = []
            for ks in statePrepare.knapsacks:
                score += ks.score_ratio()
                remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
            scores.append(score)
            remain_cap_ratios.append(np.mean(remain_cap_ratio))
        return scores, remain_cap_ratios
    
    
    
    
    