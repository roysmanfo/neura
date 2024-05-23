import random as _random
from typing import Iterable, Optional

from nnfs.layers import Layer, Input

class Model:
    def __init__(self, layers: Optional[list[Layer]] = None, input_shape: Optional[tuple[int, ...]] = None, name: Optional[str] = None, learning_rate: float = .001) -> None:
        """
        parameters:
        
        layers:         a list of layers to initialize the model. you can add more layers
                        by calling `add_layer()`
        input_shape:    a tuple with the shape of the input for the neural network.
                        if None, 1 is assumed
        name:           the name of the model
                    
        """
        
        if not isinstance(name, str) and not name is None:
            raise ValueError("name must be of type str or None")
        
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        
        self.layers: list[Layer] = []
        self.learning_rate = learning_rate        
        self.name = "Model" if name is None else str(name) or "Model"

        if input_shape and not all(isinstance(i, int) for i in input_shape):
            raise ValueError("input_shape can only contain int values")

        self.input_size = Input(input_shape) if input_shape else 1

        if layers:
            for layer in layers:
                if not layer:
                    raise ValueError(f"invalid layer supplied: '{layer}'")
                self.add_layer(layer)

    def add_layer(self, layer: Layer):
        """
        Adds a layer at the end of the model
        """
        if len(self.layers) > 0:
            previous_layer_node_count = len(self.layers[-1].nodes)
            n = 1
            
            while n < len(self.layers) and self.layers[-n].__class__.__name__ == "Flatten":
                n += 1

            if n == len(self.layers):
                previous_layer_node_count = self.input_size
            else:
                previous_layer_node_count = len(self.layers[-1].nodes)

            for n in layer.nodes:
                n.weights = [_random.uniform(-1, 1) for _ in range(previous_layer_node_count)]
        else:
            for n in layer.nodes:
                n.weights = [_random.uniform(-1, 1) for _ in range(self.input_size)]

        self.layers.append(layer)

    def summary(self, verbose: bool = True) -> str:
        """
        Returns a summary of the model architecture

        `:param` verbose when set to False does not print anything
        """
        params = 0

        layer_info = ""

        for i, layer in enumerate(self.layers):
            layer_info += f"{i}. {type(layer).__name__}"
            p = 0
            for node in layer.nodes:
                p += len(node.weights)
                params += len(node.weights)

            layer_info += f"\t\tnodes: {len(layer.nodes)}"
            layer_info += f"\tparams: {p}\n"

        summary = f"""
{self.name}
===============================================
Total number of parameters: {params}
===============================================

LAYERS:

{layer_info}
===============================================
"""
        if verbose:
            print(summary)
        return summary

    def predict(self, values: Iterable[float], verbose: Optional[bool] = True) -> list[float]:
        values = list(values)
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"predicting (layer: {i + 1} / {len(self.layers)})", end="\r")
            values = layer.calc(values)
        if verbose:
            print(f"predicting (layer: {len(self.layers)} / {len(self.layers)})")
        return values
