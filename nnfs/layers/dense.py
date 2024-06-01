import numpy as np

from nnfs.layers import Layer
from nnfs.nodes import Node
from nnfs.utils.types import InputValue, OutputValue


class Dense(Layer):
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Layer):
            return False
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    def _analize(self, x: InputValue, node: Node) -> float:
        res = node.calc(x) + self.bias
        return res

    def calc(self, x: InputValue) -> OutputValue:    
        if not isinstance(x, (np.ndarray)):
            raise ValueError("incompatible type: expected np.ndarray, received:", type(x).__name__)

        v: OutputValue = np.array([], dtype=x.dtype)

        for node in self.nodes:
            v = np.append(v, self._analize(x, node))

        return v
