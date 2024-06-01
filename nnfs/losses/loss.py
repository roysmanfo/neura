import numpy as np
import math as _math
from abc import ABC, abstractmethod

from nnfs.utils.types import InputValue



class Loss(ABC):
    """
    Base class for all loss functions

    All classes that inherit this class have to redefine the method `call()`
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def call(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        ...


class MeanAbsoluteError(Loss):
    """
    L(a) = |a|
    """

    def call(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        return np.mean(np.abs(y_true - y_pred))



class MeanSquaredError(Loss):
    """
    L(a) = a^2
    """

    def call(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        return np.mean((y_true - y_pred) ** 2)
    
class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss function (Log Loss)
    
    This is mostly used for categorical models
    """
    
    def call(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        # ? avoid calculating log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) 

class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss function
    
    This is mostly used for categorical models
    """

    def call(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        # ? avoid calculating log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred))
    