import random
import numpy as np
import torch

def fix_seed_all(seed: int) -> None:
    '''
    Fix seed for experiment reproduction. (random, numpy, torch)
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return None