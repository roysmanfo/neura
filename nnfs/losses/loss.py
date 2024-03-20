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
    