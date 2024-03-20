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
    def call(self, expected_values: List[float], predicted_values: List[float]) -> float:
        ...
    