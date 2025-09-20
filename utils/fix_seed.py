import random
import numpy as np
import torch

def fix_seed(seed: int = 42, mode: str = 'all') -> None:
    '''
    Fix seed for experiment reproduction. (random, numpy, torch)

    Parameters:
        seed: Seed number. Default 42.
        mode: Which library that the seed fixing will be applied. Default 'all'
    '''
    if mode == 'random':
        random.seed(seed)
    elif mode == 'numpy':
        np.random.seed(seed)
    elif mode== 'torch':
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    elif mode == 'all':
        random.seed(seed)
        np.random.seed(seed)
        torch.random.seed(seed)
        torch.backends.cudnn.deterministic = True
    return None