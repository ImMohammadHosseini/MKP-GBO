
"""

"""
import numpy as np
from numpy import unravel_index
from dataclasses import dataclass
from typing import List
from .knapsack import Knapsack
from typing import List, Optional
import math

@dataclass
class StatePrepare:
    knapsacks: List[Knapsack]
    
    remainInstanceWeights: np.ndarray
    remainInstanceValues: np.ndarray
    
    pickedWeightsValues: List[int] = None
    
    def __init__ (
        self, 
        info: dict,
        allCapacities: np.ndarray, 
        weights: np.ndarray, 
        values: np.ndarray, 
        k_obs_size: Optional[int] = None, 
        i_obs_size: Optional[int] = None
    ) -> None:
        assert len(weights) == len(values)        
        
        self.info = info
        
        self.weights = weights
        self.values = values
        
        self.caps = allCapacities

        if k_obs_size == None: 
            self.knapsackObsSize = len(allCapacities)
        else: self.knapsackObsSize = k_obs_size
        
        if i_obs_size == None:
            self.instanceObsSize = len(self.weights)
        else: self.instanceObsSize = i_obs_size
        
        self.allocated_matrix = None
        self.pad_len = 0
        
    def reset (self) -> None:
        self.allocated_matrix = None
        
    def _setKnapsack (self, allCapacities):
        self.knapsacks = [Knapsack(i, c) for i, c in enumerate(allCapacities)]
    
    def getObservation (self) -> np.ndarray:
        if self.allocated_matrix == None:
            ks_rep = math.ceil(len(self.weights)/len(self.caps))
            row = ks_rep * len(self.caps)
            alloc_part = np.array([[0]*len(self.caps)+[1]]*row)
            inst_pad = np.array([[0]*(len(self.weights[0])+1)]*(row-len(self.weights)))
            self.allocated_matrix = np.append(np.append(np.append(np.append(self.weights, 
                                                                            self.values,1),
                                                                  inst_pad,0),
                                                        np.repeat(self.caps,ks_rep, 0),1),alloc_part,1)
        
        return self.allocated_matrix
        
        
    def getKnapsack (self, index) -> Knapsack:
        return self.caps[index]
    
    def getObservedInst (self, index):
        return self.stateWeightValues[index]
    
    def getInstWeight (self, index):
        return self.weights[index]
    
    def getInstValue (self, index):
        return self.values[index]
    
    def is_terminated (self):
        pass
    
    def updateMatrix (self, acts):
        pass