from easydl import *
import numpy as np

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def normalize_weight(x, cut, expend=True, numpy=False):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val + 1e-10)
    if expend:
        mean_val = torch.mean(x)
        x = x / (mean_val + 1e-10)
        x = torch.where(x >= cut, x, torch.zeros_like(x))
    if numpy:
        x = variable_to_numpy(x)
    return x.detach()


def l2_norm(input, dim=1):
    norm = torch.norm(input,dim=dim,keepdim=True)
    output = torch.div(input, norm)
    return output


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
