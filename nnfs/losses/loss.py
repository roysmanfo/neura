import math as _math
from typing import List
from abc import ABC, abstractmethod

class Loss(ABC):
    """
    Base class for all loss functions

    All classes that inherit this class have to redefine the method `call()`
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        ...


class MeanAbsoluteError(Loss):
    """
    L(a) = |a|
    """

    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        s = sum([a - b for a, b in zip(y_true, y_pred)])
        return abs(s)


class MeanSquaredError(Loss):
    """
    L(a) = a^2
    """

    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        s = sum([(a - b)**2 for a, b in zip(y_true, y_pred)])
        return s / len(y_true)
    
class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss function (Log Loss)
    
    This is mostly used for categorical models
    """
    
    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        s = sum([( y * _math.log(y_p) + (1 - y) * _math.log( 1 - y_p)) for y, y_p in zip(y_true, y_pred)])
        return -s / len(y_pred) 

class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss function
    
    This is mostly used for categorical models
    """

    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        s = sum([( y * _math.log(y_p)) for y, y_p in zip(y_true, y_pred)])
        return -s / len(y_pred) 
    