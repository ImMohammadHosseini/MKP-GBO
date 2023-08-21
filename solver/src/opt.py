
"""

"""
import torch
from copy import deepcopy


def convertToOneHot(dat, fixed_part, ksNum):
    alloc = []
    for i in dat:
        oneHot = [0] * (ksNum+1); alist = i.tolist()[-(ksNum+1):]
        oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
    new_dat_oneHot = torch.cat((fixed_part, torch.FloatTensor(alloc)), dim=1)
    return new_dat_oneHot

def opt(observed, model, ksNum):
    observed.requires_grad = True
    optimizer = torch.optim.Adam([observed] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0
    fixed_part = deepcopy(observed.data[:,0:-(ksNum+1)])
    while iteration < 200:
        alloc_old = deepcopy(observed.data[:,-(ksNum+1):])
        z = model(observed)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        observed.data = convertToOneHot(observed.data, fixed_part, ksNum)
        equal = equal + 1 if torch.all(alloc_old.eq(observed.data[:,-(ksNum+1):])) else 0
        if equal > 30: break
        iteration += 1
    observed.requires_grad = False 
