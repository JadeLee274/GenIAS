from utils.common_import import *


def f1(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall)