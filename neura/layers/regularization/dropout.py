import warnings

import numpy as np

from neura.layers import Layer
from neura.utils.types import InputValue, OutputValue

class Dropout(Layer):

    def __init__(self, rate: np.float64, seed: int | None = None):
        super().__init__(
            units=1,
            bias=False,
            activation=None,
        )
        

        self.rate = rate
        self.seed = seed
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self.trainable = False

        warnings.warn("this layer is still unstable (for some reason)", RuntimeWarning)

    def forward(self, x: InputValue) -> OutputValue:
        if not isinstance(x, (np.ndarray)):
            raise ValueError("incompatible type: expected np.ndarray, received:", type(x).__name__)

        self.input = x
        node_probability = self.generator.normal(0, 1, x.size)
        self.outputs = np.where(node_probability < self.rate, x * 0, x / (1 - self.rate))
        print(self.outputs, self.input)
        return self.outputs

