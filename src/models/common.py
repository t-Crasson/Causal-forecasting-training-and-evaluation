import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np


def create_sequential(nn_final_m, hidden_size,tgt_size):
    nn_final = copy.deepcopy(nn_final_m)
    nn_final.append(tgt_size)
    last_nn = nn.Sequential()
    last_nn.append(nn.Linear(in_features = hidden_size, out_features = nn_final[0]))
    for i in range(len(nn_final) - 1):
        last_nn.append(nn.ReLU())
        last_nn.append(nn.Linear(in_features = nn_final[i], out_features = nn_final[i+1]))
   
    return last_nn

def sample_t(x,proj_len):
    return np.random.randint(1,x-proj_len-1)
vectorize_sample_t = np.vectorize(sample_t)