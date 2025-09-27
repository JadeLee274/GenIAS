import random
import logging
from math import cos, pi
import torch.optim as optim
from .common_import import *


def str2bool(v: str) -> bool:
    """
    Changes string to bool.

    Parameters:
        v: String. Must be either 'True' or 'False'.
    """
    assert v in ['True', 'False'], "string must be either 'True' or 'False'"

    return v.lower() in 'true'


def cosine_schedule(
    optimizer: optim.Adam,
    current_epoch: int,
    total_epochs: int = 30,
    initial_learning_rate: float = 1e-3,
    lr_decay_rate: float = 0.01,
) -> None:
    """
    Customized cosine scheduler. Updates optimizer's learning rate.

    Parameters:
        optimizer:             Adam.
        current_epoch:         Current training epoch.
        total_epochs:          Total training epochs. Default 30.
        initial_learning_rate: Initial learning rate. Defalut 1e-3.
        lr_dacay_rate:         Decay rate of initial learning rate.
                               Default 0.01.
    """
    eta_min = initial_learning_rate * (lr_decay_rate ** 3)
    scheduled_learning_rate = eta_min \
    + (initial_learning_rate - eta_min) \
    * (1 + cos(pi * current_epoch / total_epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduled_learning_rate

    return


def set_logging_filehandler(
    log_file_path: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
) -> None:
    '''
    Set logging FileHandler + StreamHandler

    Parameters:
        log_file_path: The directory + filename of log file. The os library
                       will make the directory where the file will be saved.
        mode:          The mode that specifies how the log file is opened.
                       Default 'w'.
        encoding:      The name of encoding that is used to encode or decode 
                       file. Default 'utf-8'.
    '''
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    format = '%(asctime)s %(message)s'
    logging.basicConfig(
        format=format,
        handlers=[
            logging.FileHandler(
                log_file_path,
                mode=mode,
                encoding=encoding,
            ),
            logging.StreamHandler(),
        ],
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logging.info(f'Logging (File + Stream) Initialized')

    return


def set_logging_streamhandler() -> None:
    '''
    Set logging StreamHandler
    '''
    logging.basicConfig(
        format=format,
        handlers=[
            logging.StreamHandler(),
        ],
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logging.info(f'Logging (Stream) Initialized')

    return


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
    
    return