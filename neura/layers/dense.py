import numpy as np

from neura.layers import Layer
from neura.nodes import Node
from neura.optimizers.base import Optimizer
from neura.utils.types import Gradients, InputValue, OutputValue


class Dense(Layer):
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Layer):
            return False
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    def _analize(self, x: InputValue, node: Node) -> np.float64:
        res = node.calc(x) + self.bias
        return res

    def forward(self, x: InputValue) -> OutputValue:    
        if not isinstance(x, (np.ndarray)):
            raise ValueError("incompatible type: expected np.ndarray, received:", type(x).__name__)

        self.input = x
        self.outputs = np.array([node.calc(x) for node in self.nodes])
        if self.bias:
            self.outputs += self.bias
        return self.outputs

