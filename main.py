
"""

"""
import optparse
import torch
import math
import numpy as np
from numpy import genfromtxt
from os import path, makedirs
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.dataProducer import multiObjectiveDimentional

from solver.src.data_structure.state_prepare_ import StatePrepare
from solver.src.models.mlp import MLP
from solver.gbo import GBO

usage = "usage: python main.py -D <dim> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-D", "--dim", action="store", dest="dim", default=1)
parser.add_option("-K", "--knapsaks", action="store", dest="kps", default=2)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=15)
parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='train')

opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':80,
         'CAP_HIGH':350, 
         'WEIGHT_LOW':10, 
         'WEIGHT_HIGH':100,
         'VALUE_LOW':3, 
         'VALUE_HIGH':200}

KNAPSACK_NUM = opts.kps
INSTANCE_NUM = opts.instances
N_TRAIN_STEPS = 20000
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models'
DATA_PATH = 'dataset/'
TRAIN_ALGORITHMS = ['SAC']

def dataInitializer ():
    if path.exists(DATA_PATH):
        instance_main_data = genfromtxt(DATA_PATH+'instances.csv', delimiter=',')
        w = instance_main_data[:,:-1]
        v = np.expand_dims(instance_main_data[:,-1],1)
        ks_main_data = genfromtxt(DATA_PATH+'ks.csv', delimiter=',')
        c = ks_main_data[:,:-1]
    else:
        c, w, v = multiObjectiveDimentional(opts.dim, opts.kps, opts.instances, INFOS)
        
    if not path.exists(DATA_PATH):
        instance_main_data = np.append(w,v,1)
        ks_main_data = np.append(c, np.zeros((c.shape[0],1)),1)
        makedirs(DATA_PATH)
        np.savetxt(DATA_PATH+'instances.csv', instance_main_data, delimiter=",")
        np.savetxt(DATA_PATH+'ks.csv', ks_main_data, delimiter=",")

    statePrepare = StatePrepare(INFOS, c, w, v, KNAPSACK_NUM, 
                                        INSTANCE_NUM)
    return statePrepare

def init(statePrepare):
    row = math.ceil(INSTANCE_NUM/KNAPSACK_NUM) * KNAPSACK_NUM
    model = MLP(row, 2*opts.dim+KNAPSACK_NUM+2) 
    solver = GBO(model,statePrepare, KNAPSACK_NUM, SAVE_PATH, True)
    
    return solver
def solve_step ():
    solver.run_GBO()
    
if __name__ == '__main__':
    statePrepare = dataInitializer()
    solver = init(statePrepare)
    solve_step()