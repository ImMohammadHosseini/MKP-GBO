
"""

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np


def multiObjectiveDimentional (D:int, M:int, N:int, info):
    """
    multi-Objective multi-Dimentional multipleKnapSack, 
    multi capacities for each knapsack, 
    multi weight and one value for each instance
    set a random dataset for M knapsack and N instance
    produce knapsack capacity array in size of (d,M), instance weight array 
    in size of (d,N)  and instance value array in size of N
    """
    capacities = np.random.randint(low=info['CAP_LOW'], high=info['CAP_HIGH'], 
                                   size=(M,D))
    weights = np.random.randint(low=info['WEIGHT_LOW'], high=info['WEIGHT_HIGH'], 
                                size=(N,D))
    values = np.random.randint(low=info['VALUE_LOW'], high=info['VALUE_HIGH'], 
                               size=(N,1))
    return capacities, weights, values