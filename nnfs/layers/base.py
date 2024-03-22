from typing import Iterable, List, Optional, Union
from abc import ABC as _ABC, abstractmethod as _abstractmethod

import nnfs.activation as _activation
from nnfs.nodes import Node
from nnfs.losses.loss import Loss as _Loss



class Layer(_ABC):
    """
    Abstract Base Class for all layers
    """

    def __init__(self, units: int, bias: Optional[int] = None, activation: Optional[Union[str, _activation.ActivationFunction]] = None) -> None:
        self._last_layer = False
        self.all_input_at_once = False
        self.loss = None

        if units < 1:
            raise ValueError("Invalid number of nodes: units < 1")
        
        if isinstance(activation, _activation.ActivationFunction):
            self.activation = activation
        
        elif isinstance(activation, str):
            match activation:
                case "sigmoid": self.activation = _activation.Sigmoid()
                case "relu": self.activation = _activation.ReLu()
                case "leakyrelu": self.activation = _activation.LeakyReLu()
                case "derivative": self.activation = _activation.Derivative()
                case "tanh": self.activation = _activation.Tanh()
                case "linear": self.activation = _activation.Linear()
                case _: raise ValueError(f"Invaid type for activation: '{activation}'")
        
        elif not activation:
            self.activation = _activation.Linear()
            
        else:
            raise ValueError(f"Invaid type activation: {type(activation)} is not (str, ActivationFunction, None)")

        self.nodes: list[Node] = [Node(self.activation) for _ in range(units)]
        self.bias = bias if bias else 0

    
    def __str__(self) -> str:
        return type(__class__).__name__ + f"(nodes={len(self.nodes)}, bias={self.bias})"
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Layer):
            return False
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    def is_last_layer(self) -> bool:
        return self._last_layer

    def set_is_last_layer(self, val: bool) -> None:
        self._last_layer = val

    
    def _analize(self, x: float, conn_n: Optional[int] = None) -> float:
        """
        Evaluate the values to pass to given next node

        y = b + ∑(wx)
        """
        ...

    @_abstractmethod
    def calc(self, x: Union[float, Iterable[float]], next_nodes: Optional[list[Node]]) -> List[float]:
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