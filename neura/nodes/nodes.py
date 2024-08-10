import numpy as np

from abc import ABC as _ABC, abstractmethod as _abstractmethod

import neura.activation as _activation
from neura.utils.types import Gradients, InputValue, NodeOutput, NodeWeights, NodeWeight


class BaseNode(_ABC):
    def __init__(self, activation: _activation.ScalarFunction) -> None:
        self.weights: NodeWeights = np.array([], dtype=NodeWeight)
        self.activation = activation
        
    def __str__(self) -> str:
        return f"{type(self).__name__}(activation='{self.activation.name}', weights={self.weights})"

    def activate(self, res: np.float64) -> np.float64:
        return self.activation.apply_formula(res)

    @_abstractmethod
    def calc(self, x: InputValue) -> NodeOutput:
        """
        Calculate the value of the data after it passed through this node
        """
        ...

class Node(BaseNode):
    def __init__(self, activation: _activation.ScalarFunction) -> None:
        super().__init__(activation)

        # self.input    stores input values during the forward pass
        # self.z        stores the weighted sum before activation


    def calc(self, x: InputValue) -> NodeOutput:        
        if len(x) != len(self.weights):
            raise ValueError(f"Input size {len(x)} does not match number of weights {len(self.weights)}")

        self.input = np.array(x)
        weighted_sum = np.sum(self.input * self.weights, dtype=np.float64)
        
        # Clip weighted_sum to avoid overflow
        weighted_sum = np.clip(weighted_sum, -1e10, 1e10)
        
        self.z = weighted_sum
        out = self.activate(weighted_sum)
        return np.float64(out)
    
    def compute_gradient(self, output_gradient: np.float64) -> Gradients:
        activation_derivative = self.activation.derivative(self.z)
        weight_gradient = output_gradient * activation_derivative * self.input
        return weight_gradient