

from typing import Any, Literal
from neura.layers.base import Layer
from neura.utils.types import InputValue, OutputValue

class Reshape(Layer):
    """
    Reshape the input tensor into the `new_shape` (if possible).
    """

    def __init__(self,
                 new_shape: tuple[int, ...],
                 **kwargs: dict[str, Any]
                 ) -> None:
        super().__init__(
            units=1,
            bias=False,
            activation=None,
            input_shape=None,
            **kwargs
            )
        
        self.new_shape = new_shape

        self.trainable = False
        self.all_input_at_once = True
        # self.pass_trough_layer = True
        self.output_shape = new_shape

        self._validate_args()

    def _validate_args(self) -> None:
        if not isinstance(self.new_shape, tuple) or not "".join([str(i) for i in self.new_shape]).isnumeric():
            raise ValueError("new shape must be a tuple of ints")
        
    def forward(self, x: InputValue) -> OutputValue:
        return x.reshape(self.new_shape)
