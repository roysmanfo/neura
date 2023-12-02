from typing import Iterable, Optional
from nnfs.layers import Layer

from nnfs.nodes import Node

class Dense(Layer):
    def __str__(self) -> str:
        return f"Dense(nodes={len(self.nodes)}, bias={self.bias})"
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Layer):
            return False
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    def _analize(self, x: float, conn_n: Optional[int] = None) -> float:
        return sum([node.calc(x, conn_n) for node in self.nodes]) + self.bias

    def calc(self, x: float, next_nodes: Optional[list[Node]]) -> Iterable[float]:
        v = []
        if not self._last_layer and next_nodes:
            for i, _ in enumerate(next_nodes):
                v.append(self._analize(x, i))
        else:
            for i, _ in enumerate(self.nodes):
                v.append(self._analize(x))


        return v

