from abc import ABC as _ABC, abstractmethod as _abstractmethod
import neura.activation as _activation
from neura.utils.types import InputValue

class BaseNode(_ABC):
    def __init__(self, activation: _activation.ScalarFunction) -> None:
        self.weights: list[float] = []
        self.activation = activation
        
    def __str__(self) -> str:
        return f"{type(self).__name__}(activation='{self.activation.name}', weights={self.weights})"

    def activate(self, res: float) -> float:
        return self.activation.apply_formula(res)

    @_abstractmethod
    def calc(self, x: InputValue) -> float:
        """
        Calculate the value of the data after it passed through this node
        """
        ...

class Node(BaseNode):
    def calc(self, x: InputValue) -> float:        
        if len(x) != len(self.weights):
            raise ValueError(f"Input size {len(x)} does not match number of weights {len(self.weights)}")
        weighted_sum = sum(i * w for i, w in zip(x, self.weights))
        return self.activate(weighted_sum)
