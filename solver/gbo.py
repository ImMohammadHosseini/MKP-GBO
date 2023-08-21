
"""

"""


from .src.models.train import load_model, save_model, sac_train
from .src.data_structure.state_prepare_ import StatePrepare
from .src.opt import opt
import torch

class GBO():
    def __init__(
        self,
        model,
        instNum: int,
        ksNum: int,
        save_path:str, 
        train_flag: bool = False,
    ):
        self.save_path = save_path
        self.instNum = instNum
        self.ksNum = ksNum
        self.model = model
        self.model = load_model(self.model, self.save_path)
        
    def run_GBO(
        self,
        observed_tensor
    ):
        opt(observed_tensor, self.model, self.ksNum)
        decision = []
        for i, row in enumerate(observed_tensor): 
            decision_part = row[-(self.ksNum+1):].tolist()
            if i == self.instNum:
                break
            if decision_part[-1] == 1:
                continue
            decision.append((i, decision_part.index(max(decision_part))))
        return decision
