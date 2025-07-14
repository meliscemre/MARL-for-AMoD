import numpy as np
import torch.nn as nn
from copy import deepcopy


def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

def nestdictsum(dict):
    return sum([sum([dict[i][t] for t in dict[i]]) for i in dict])

def moving_average(a, n=3) :
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def set_requires_grad_flag(net: nn.Module, requires_grad: bool) -> None:
    for p in net.parameters():
        p.requires_grad = requires_grad

def create_target(net: nn.Module) -> nn.Module:
    target = deepcopy(net)
    set_requires_grad_flag(target, False)
    return target