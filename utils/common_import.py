"""
Imports commomly used libraries, modules, and alias.
"""
from typing import *
import os
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
Vector = np.ndarray
Matrix = np.ndarray
Array = np.ndarray

def str2bool(v: str) -> bool:
    """
    Converts string (Either True or False) to bool.

    Parameters:
        v: String instance. Either 'True' or 'False.'

    Returns:
        Boolean instance. Either True or False.
    """
    return v.lower() in ('true')