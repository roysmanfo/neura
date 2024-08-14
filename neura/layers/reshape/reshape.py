

from typing import Any, Literal
from neura.layers.base import Layer
from neura.utils.types import InputValue, OutputValue

class Reshape(Layer):
    """
    Reshape the input tensor into the `new_shape` (if possible).

    When there aren't enough values, and `pad` is True, the layer will  
    add additional zeros in order to match the needed shape,  
    and `order` will determine where

    - start: The zeros will be added at the start of the tensor
    - end: The zeros will be added at the end of the tensor
    - even: The zeros will be added evently between start and end
    
    """

    def __init__(self,
                 new_shape: tuple[int, ...],
                 pad: bool = False,
                 order: Literal["start", "end", "even"] = "end",
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
        self.pad = pad
        self.order = order

        self.trainable = False
        self.all_input_at_once = True
        # self.pass_trough_layer = True
        self.output_shape = new_shape

        self._validate_args()

    def _validate_args(self) -> None:
        if not isinstance(self.new_shape, tuple) or not "".join([str(i) for i in self.new_shape]).isnumeric():
            raise ValueError("new shape must be a tuple of ints")
        
        if not isinstance(self.pad, bool):
            raise ValueError("pad must be a bool")
        
        if not isinstance(self.order, str) or not self.order in ("start", "end", "even"):
            raise ValueError("order must be either 'start', 'end' or 'even' ")

    def forward(self, x: InputValue) -> OutputValue:
        return x.reshape(self.new_shape)
