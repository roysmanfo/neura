from typing import Any, Iterable, List, Optional, Union
from nnfs.layers import Layer


class Flatten(Layer):
    def __init__(self, input_shape: Optional[tuple[int, ...]] = None) -> None:
        super().__init__(
            units=1,
            bias=False,
            activation=None,
            input_shape=input_shape
        )
        
        self.all_input_at_once = True

    def calc(self, x: Union[float, Iterable[float]]) -> List[float]:
        if isinstance(x, (int, float, complex)):
            raise ValueError("expected Iterable, received:", type(x).__name__)

        return self._unpack(x)

    def _unpack(self, paked: Any) -> list[float]:
        vals: list[float] = []
        for item in paked:
            if isinstance(item, list):
                vals.extend(self._unpack(item))
            else:
                vals.append(item)
        return vals
