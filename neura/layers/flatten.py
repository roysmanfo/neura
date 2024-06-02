import numpy as np
from typing import Any, Optional

from neura.layers import Layer
from neura.utils.types import InputValue, OutputValue


class Flatten(Layer):
    def __init__(self, input_shape: Optional[tuple[int, ...]] = None) -> None:
        super().__init__(
            units=1,
            bias=False,
            activation=None,
            input_shape=input_shape
        )
        
        self.all_input_at_once = True

    def calc(self, x: InputValue) -> OutputValue:
        if isinstance(x, (int, float, complex)):
            raise ValueError("expected Iterable, received:", type(x).__name__)

        return self._unpack(x)

    def _unpack(self, paked: Any) -> OutputValue:
        vals = np.array([], dtype=np.float64)

        for item in paked:
            if isinstance(item, np.ndarray):
                vals = np.concatenate([vals, self._unpack(item)])
            else:
                vals = np.append(vals, item)
        return vals
