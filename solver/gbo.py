
"""

"""
import sys
sys.path.append('scheduler/BaGTI/')

from .src.models.train import load_model, save_model, sac_train
from .src.data_structure.state_prepare_ import StatePrepare
from .src.opt import opt
import torch

class GBO():
    def __init__(
        self,
        model,
        statePrepare: StatePrepare,
        ksNum: int,
        save_path:str, 
        train_flag: bool = False,
    ):
        self.save_path = save_path
        self.ksNum = ksNum
        self.statePrepare = statePrepare
        self.model = model
        self.model = load_model(self.model, self.save_path)
        if train_flag:
            sac_train(self.model)
        
    def run_GBO(self):
        observed_tensor = torch.tensor(self.statePrepare.getObservation()).float()
        result, iteration, fitness = opt(observed_tensor, self.model, self.ksNum)
        decision = []
        for cid in observed_tensor: 
            one_hot = result[cid, -(self.ksNum+1):].tolist()  
            new_host = one_hot.index(max(one_hot))
            #if prev_alloc[cid] != new_host: decision.append((cid, new_host))
            return decision
