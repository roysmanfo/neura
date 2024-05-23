from typing import Any, Iterable, List, Union
from nnfs.layers import Layer


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__(1, 0, None)
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
