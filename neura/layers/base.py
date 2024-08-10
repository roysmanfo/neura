from typing import Optional, Union
from abc import ABC as _ABC, abstractmethod as _abstractmethod
import random as _random
import numpy as np

import neura.activation as _activation
from neura.nodes import Node
from neura.losses.loss import Loss as _Loss
from neura.utils.types import InputValue, OutputValue

Activation = Union[str, _activation.Activation]


class Layer(_ABC):
    """
    Abstract Base Class for all layers

    parameters:

    - units:          the number of perceptrons for the current layer
    - bias:           if true, a bias will be added to the output 
    - activation:     the function used to fire each perceptron of the layer
    - input_shape:    a tuple with the shape of the input for the neural network.
                      if None, then 1 is assumed
    """

    def __init__(self,
                 units: int,
                 bias: Optional[bool] = None,
                 activation: Optional[Activation] = None,
                 input_shape: Optional[tuple[int, ...]] = None

                 ) -> None:
        
        self._last_layer = False
        self.all_input_at_once = False
        self.loss = None

        if units < 1:
            raise ValueError("Invalid number of nodes: units < 1")
        
        if isinstance(activation, _activation.Activation):
            self.activation = activation
        
        elif isinstance(activation, str):
            self.activation = self._get_activation_func(activation)
            if self.activation is None:
                raise ValueError(f"Invaid type for activation: '{activation}'")
        
        elif not activation:
            self.activation = _activation.Linear()
            
        else:
            raise ValueError(f"Invaid type activation: {type(activation)} is not (str, Activation, None)")
        
        self.nodes: list[Node] = [Node(self.activation if isinstance(self.activation, _activation.ScalarFunction) else _activation.Linear()) for _ in range(units)]
        self.bias = _random.gauss(mu=0, sigma=1) if bias else 0

        if input_shape:
            if not isinstance(input_shape, tuple):
                raise ValueError("input_shape must be a tuple of int", type(input_shape))
            
            if not all(i > 0 for i in input_shape):
                raise ValueError("input_shape numbers must be > 0")

        self.input_shape = input_shape


    #! temporary
    def _get_activation_func(self, name: str) -> _activation.Activation | None:
        func = {
            "exponential":  _activation.Exponential(),
            "leakyrelu":    _activation.LeakyReLu(),
            "linear":       _activation.Linear(),
            "prelu":        _activation.PReLU(),
            "relu":         _activation.ReLu(),
            "sigmoid":      _activation.Sigmoid(),
            "softmax":      _activation.Softmax(),
            "swish":        _activation.Swish(),
            "tanh":         _activation.Tanh(),
        }

        return func.get(name.lower(), None)
    
    def __str__(self) -> str:
        return self.__class__.__name__ + f"(nodes={len(self.nodes)}, bias={self.bias})"
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Layer):
            return False
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    @property
    def is_last_layer(self) -> bool:
        return self._last_layer

    def set_is_last_layer(self, val: bool) -> None:
        self._last_layer = val

    
    def _analize(self, x: InputValue, node: Node) -> float:
        """
        Evaluate the values to pass to given next node

        y = b + ∑(wx)
        """
        ...

    @_abstractmethod
    def calc(self, x: InputValue) -> OutputValue:
        """
        Evaluate the values to pass to the next layer/output
        """
        ...

    def add_loss(self, func: _Loss) -> None:        
        """
        Add a loss function to the current layer 
        """
        if not issubclass(type(func), _Loss):
            raise ValueError("unsupported loss function of type: %s " % type(func))
        
        self.loss = func