from typing import Any
import numpy as np

from neura.layers import Layer
from neura.layers.exceptional import NotFirstLayer
from neura.utils.types import InputValue, OutputValue

class Dropout(Layer, NotFirstLayer):

    def __init__(self,
                 rate: float | np.float64,
                 seed: int | None = None,
                 input_shape: tuple[int, ...] | None = None,
                 **kwargs: dict[str, Any]
                 ) -> None:
        super().__init__(
            units=1,
            bias=False,
            activation=None,
            input_shape=input_shape,
            **kwargs
            )

        self.rate = np.float64(rate)
        self.seed = seed
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self.trainable = False

    def forward(self, x: InputValue) -> OutputValue:
        if not isinstance(x, (np.ndarray)):
            raise ValueError("incompatible type: expected np.ndarray, received:", type(x).__name__)

        self.input = x

        # outside training this layer becomes transparent
        if not self.training:
            self.outputs = x
            return x

        node_probability = self.generator.normal(0, 1, x.size).reshape(x.shape)
        self.outputs = np.where(node_probability < self.rate, x * 0, x / (1 - self.rate))
        return self.outputs

