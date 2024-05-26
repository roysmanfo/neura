from typing import List
from nnfs.layers import Layer
from nnfs.nodes import Node
from nnfs.utils.types import InputValue

class Dense(Layer):
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Layer):
            return False
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    def _analize(self, x: List[float], node: Node) -> float:
        res = node.calc(x) + self.bias
        return res

    def calc(self, x: InputValue) -> List[float]:    
        if not isinstance(x, (list, tuple)):
            raise ValueError("incompatible type: expected list or tuple, received:", type(x).__name__)

        v: List[float] = []

        for node in self.nodes:
            v.append(self._analize(x, node)) # type: ignore

        return v
