import random
from typing import Iterable, Optional, Union
from activation import ActivationFunction, Basic, Sigmoid, ReLu, LeakyReLu, Tanh, Derivative
from numpy import mean

class Node:
    def __init__(self, activation: Optional[Union[ActivationFunction, str]] = None) -> None:
        self.weights: list[float] = []
        self.w = random.random()
        self.activation = activation or Basic()

    def __str__(self) -> str:
        return f"Node(activation='{self.activation.name}', weigths={self.weights})"

    def activate(self, res: float) -> float:
        return self.activation.apply_formula(res)

    def calc(self, x: float, conn_n: Optional[int] = None):
        """
        Calculate the value of the data after it passed trough this node
        based on the destination node
        """
        if not conn_n:
            return self.activate(x * self.w)
        
        if conn_n < 0 or conn_n > len(self.weights):
            raise ValueError(f"Invalid connection_number (must be 0 < x < len(self.weights)): conn_n={conn_n}")
        return self.activate(x * self.weights[conn_n])
    


class Layer:
    def __init__(self, units: int, bias: Optional[int] = None, activation: Optional[str] = None) -> None:
        self._last_layer = False
        if units < 1:
            raise ValueError("Invalid number of nodes: units < 1")
        
        if isinstance(activation, ActivationFunction):
            self.activation = activation
        
        elif activation is None:
            self.activation = Basic()
        
        elif isinstance(activation, str):
            match activation:
                case "sigmoid": self.activation = Sigmoid()
                case "relu": self.activation = ReLu()
                case "leakyrelu": self.activation = LeakyReLu()
                case "derivative": self.activation = Derivative()
                case "tanh": self.activation = Tanh()
                case "basic": self.activation = Basic()
                case _: raise ValueError(f"Invaid type for activation: '{activation}'")
        
        else:
            raise ValueError(f"Invaid type activation: {type(activation)} is not (str, ActivationFunction, None)")
        
        self.nodes: list[Node] = [Node(self.activation) for _ in range(units)]
        self.bias = bias if bias else 0


    def __str__(self) -> str:
        return f"Layer(nodes={len(self.nodes)}, bias={self.bias})"
    
    def __eq__(self, __value: object) -> bool:
        return len(self.nodes) == len(__value.nodes) and all([node == __value.nodes[index] for index, node in enumerate(self.nodes)]) and self.bias == __value.bias

    def _analize(self, x: float, conn_n: Optional[int] = None) -> float:
        """
        Evaluate the values to pass to given next node

        y = b + ∑(wx)
        """
        return sum([node.calc(x, conn_n) for node in self.nodes]) + self.bias

    def calc(self, x: float, next_nodes: Optional[list[Node]]) -> Iterable[float]:
        """
        Evaluate the values to pass to the next layer/output
        """
        v = []
        if not self._last_layer or next_nodes:
            for i, _ in enumerate(next_nodes):
                v.append(self._analize(x, i))
        else:
            for i, _ in enumerate(self.nodes):
                v.append(self._analize(x))


        return v

class NeuralNetwork:
    def __init__(self, layers: Optional[list[Layer]], name: Optional[str] = None, learning_rate: float = .5) -> None:
        self.layers: list[Layer] = []
        self.learning_rate = learning_rate        
        self.name = str(name) or "NeuralNetwork"

        for layer in layers:
            if not isinstance(layer, Layer):
                raise ValueError(f"invalid layer supplied: '{layer}'")
            self.add_layer(layer)

    def add_layer(self, layer: Layer):
        """
        Adds a layer at the end of the model
        """
        if len(self.layers) > 0:
            self.layers[-1]._last_layer = False
            for n in self.layers[-1].nodes:
                for _ in layer.nodes:
                    n.weights.append(random.random())
        
        layer._last_layer = True
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
        res = []
        values = list(values)

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"predicting (layer: {i} / {len(self.layers)})", end="\r")

            for val in values:
                layer_output = layer.calc(val, self.layers[i + 1].nodes) if i < len(self.layers) - 1 else layer.calc(val, None)
                res.append(layer_output)

            values.clear()
            for i in range(len(res[0])):
                values.append(mean([f[i] for f in res]))
            
            res.clear()

        if verbose:
            print(f"predicting (layer: {i} / {len(self.layers)})")
       
        return values
