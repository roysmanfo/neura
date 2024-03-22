import random as _random
from typing import Optional
from abc import ABC as _ABC, abstractmethod as _abstractmethod
import nnfs.activation as _activation

class BaseNode(_ABC):
    def __init__(self, activation: Optional[_activation.ActivationFunction] = None) -> None:
        self.weights: list[float] = []
        self.w = _random.uniform(-1, 1)
        self.activation: _activation.ActivationFunction
        
        if isinstance(activation, _activation.ActivationFunction):
            self.activation = activation
        else: 
            self.activation = _activation.Linear()

    def __str__(self) -> str:
        return f"{type(self).__name__}(activation='{self.activation.name}', weigths={self.weights}, w={self.w})"

    def activate(self, res: float) -> float:
        return self.activation.apply_formula(res)

    @_abstractmethod
    def calc(self, x: float, conn_n: Optional[int] = None) -> float:
        """
        Calculate the value of the data after it passed trough this node
        based on the destination node
        """
        ...
    

class Node(BaseNode):

    def calc(self, x: float, conn_n: Optional[int] = None) -> float:
        if not conn_n:
            return self.activate(x * self.w)
        
        if conn_n < 0 or conn_n > len(self.weights):
            raise ValueError(f"Invalid connection_number (must be 0 < x < len(self.weights)): conn_n={conn_n}")
        return self.activate(x * self.weights[conn_n])
    
