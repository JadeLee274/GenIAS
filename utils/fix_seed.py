import random
import numpy as np
import torch

def fix_seed_all(seed: int = 42) -> None:
    '''
    Fix seed for experiment reproduction. (random, numpy, torch)

    Parameters:
        seed: Seed number. Default 42.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return None