import numpy as np
import os
import pickle


def read_agent_data(folder_name, env_name, hidden_dim, it = 0):
    file_name =  env_name +'_h' + str(hidden_dim) +'_iter' + str(it) +'.pkl'
    with open(os.path.join(folder_name, file_name),'rb') as f:
        data = pickle.load(f)
    return data


def read_expert_data(folder_name, env_name):
    file_name =  env_name +'.pkl'
    with open(os.path.join(folder_name, file_name),'rb') as f:
        data = pickle.load(f)
    return data


def shuffle_data(x, y):
    N = x.shape[0]
    shuffle_idx = np.arange(N)
    np.random.shuffle(shuffle_idx)

    return x[shuffle_idx], y[shuffle_idx]

