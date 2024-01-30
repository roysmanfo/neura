from typing import Any, Iterable, Union
from .layers import Layer
from nnfs.nodes import Node


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__(1, 0, None)
        self.all_input_at_once = True

    def calc(self, x: Union[float, Iterable[float]], next_nodes: list[Node] | None) -> Iterable[float]:
        def unpack(paked: Any) -> list[float]:
            vals: list[float] = []
            try:
                iter(paked)
                vals.extend(unpack(paked[0]))
            except:
                vals.append(paked)
            
            return vals

        v: list[float] = []
        
        if isinstance(x, (int, float, complex)):
            raise ValueError("expected Iterable, received:", type(x).__name__)

        x = list(x)

        for i in x:
            v.extend(unpack(i))
        print(v)

        # [[10.3, 43], [23, 354]]
        return v


