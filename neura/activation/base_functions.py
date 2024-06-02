from abc import ABC as _ABC, abstractmethod as _abstractmethod
from typing import Any

from neura.utils.types import InputValue, OutputValue

class Activation(_ABC):
    """
    Base class for all activation functions
    """
    
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @_abstractmethod
    def apply_formula(self, x: Any) -> Any:
        """
        Apply the formula of the activation function
        """
        ...
    
    @_abstractmethod
    def derivative(self, x: Any) -> Any:
        """
        Calculate the derivative in x
        """
        ...


class ParametricFunction(Activation):
    """
    Base class for an activation function that also takes parameters
    """

    def __init__(self, *args: list[str], **kwargs: dict[str, str]) -> None:
        super().__init__()
        self.params: dict[str, float] = {} # param_name: value

class ScalarFunction(Activation):
    """
    Base class for an activation function that also takes a vector (list) as input
    """
    @_abstractmethod
    def apply_formula(self, x: float) -> float: ... 
    
    @_abstractmethod
    def derivative(self, x: float) -> float: ...

class VectorialFunction(Activation):
    """
    Base class for an activation function that also takes a vector (list) as input
    """
    @_abstractmethod
    def apply_formula(self, x: InputValue) -> OutputValue: ...
    
    @_abstractmethod
    def derivative(self, x: InputValue) -> OutputValue: ...

    