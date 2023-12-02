import random as _random
from typing import Iterable, Optional
from numpy import mean as _mean

from nnfs.layers import Layer


class Model:
    def __init__(self, layers: Optional[list[Layer]] = None, name: Optional[str] = None, learning_rate: float = .5) -> None:
        self.layers: list[Layer] = []
        self.learning_rate = learning_rate        
        self.name = str(name) or "Model"
        
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
            self.layers[-1].set_is_last_layer(False)
            for n in self.layers[-1].nodes:
                for _ in layer.nodes:
                    n.weights.append(_random.random())
        
        layer.set_is_last_layer(True)
        self.layers.append(layer)


    def summary(self) -> str:
        """
        Returns a summary of the model architecture
        """
        params = 0

        layer_info = ""

        for i, layer in enumerate(self.layers):
            layer_info += f"Layer {i}"
            p = 0
            for node in layer.nodes:
                params += len(node.weights)                    
                p += len(node.weights)

                if node == layer.nodes[-1]:
                    params += len(layer.nodes)
                    p += len(layer.nodes)
            layer_info += f"\t\tnodes: {len(layer.nodes)}"
            layer_info += f"\tparams: {p}\n"
        
        return f"""
{self.name}
===============================================
Total number of parameters: {params}
===============================================

LAYERS:

{layer_info}
===============================================
"""


    def predict(self, values: Iterable[float], verbose: Optional[bool] = True) -> list[float]:
        res: list[list[float]] = []
        values = list(values)
        pred = 0
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"predicting (layer: {i} / {len(self.layers)})", end="\r")
                pred = i

            for val in values:
                layer_output = layer.calc(val, self.layers[i + 1].nodes) if i < len(self.layers) - 1 else layer.calc(val, None)
                res.append(list(layer_output))

            values.clear()
            for i in range(len(res[0])):
                values.append(_mean([f[i] for f in res]))
            
            res.clear()

        if verbose:
            print(f"predicting (layer: {pred} / {len(self.layers)})")
       
        return values
