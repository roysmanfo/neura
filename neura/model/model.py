import numpy as np
import random as _random
from typing import Any, List, Optional, Union

from neura import optimizers
from neura import losses
from neura.losses import Loss
from neura.layers import Layer, Input
from neura.utils.types import InputValue, OutputValue
from neura.evaluation import Evaluation


class Model:
    def __init__(self, layers: Optional[list[Layer]] = None, name: Optional[str] = None, learning_rate: float = .001) -> None:
        """
        parameters:
        
        - layers:         a list of layers to initialize the model. you can add more layers
                          by calling `add_layer()`
        - name:           the name of the model
                    
        """
        
        if not isinstance(name, str) and not name is None:
            raise ValueError("name must be of type str or None")
        
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        
        self.loss: Loss = losses.MeanSquaredError()
        self.optimizer: optimizers.Optimizer = optimizers.SGD()
        self.metrics = []

        self.layers: list[Layer] = []
        self.learning_rate = learning_rate        
        self.name = "Model" if name is None else str(name) or "Model"

        if layers:
            input_shape = layers[0].input_shape

            if input_shape is None:
                raise ValueError("no input_shape provided for the first layer")
       
            if input_shape and not all(isinstance(i, int) for i in input_shape):
                raise ValueError("input_shape can only contain int values")

            self.input_size = Input(input_shape) if input_shape else 1

            if layers:
                for layer in layers:
                    if not layer:
                        raise ValueError(f"invalid layer supplied: '{layer}'")
                    self.add_layer(layer)
        else:
            input_shape = None
            self.input_size = None



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
                n.weights = [_random.uniform(-1, 1) for _ in range(previous_layer_node_count or 1)]
        else:

            if not layer.input_shape:
                raise ValueError("no input_shape provided for the first layer")
            
            if layer.input_shape and not all(isinstance(i, int) for i in layer.input_shape):
                raise ValueError("input_shape can only contain int values")

            self.input_size = Input(layer.input_shape)
            for n in layer.nodes:
                n.weights = [_random.uniform(-1, 1) for _ in range(self.input_size)]

        self.layers.append(layer)

    def compile(self, loss: Union[Loss, str], optimizer: Optional[optimizers.Optimizer] = None, metrics: Optional[list[Any]] = None):
        assert isinstance(loss, Loss), "`loss` must be a loss function"
        assert isinstance(optimizer, optimizers.Optimizer), "`optimizer` must be a valid Optimizer instance"

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics


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

    def predict(self, values: InputValue, verbose: Optional[bool] = True) -> OutputValue:
        # TODO: add support for batch input 

        # pass all the through all layers of the network
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"predicting (layer: {i + 1} / {len(self.layers)})", end="\r")
            values = layer.calc(values)

        if verbose:
            print(f"predicting (layer: {len(self.layers)} / {len(self.layers)})")
        
        return values
    
    def evaluate(self, x: InputValue, y: InputValue) -> List[Any]:
        """
        Evaluate the models performance
        
        Parameters
            :param x (np.ndarray): a batch of sample data for the model evaluation 
            :param y (np.ndarray): the corresponding expected results 
        """
        

        if not self.loss:
            raise RuntimeError("You need to call compile() before evaluating")
        
        if not self.layers:
            raise RuntimeError("the model has no layers yet")
        
        if (input_shape := self.layers[0].input_shape) and len(input_shape) + 1 != len(x.shape):
            raise ValueError("unable to process batch input_shape {}." \
                             " expected shape: {}".format(x.shape, '(n, ' + ', '.join(str(i) for i in input_shape) + ')'))

        
        loss = 0
        for i, sample in enumerate(x):
            # TODO: reduce overhead and increase efficiency by processing all the data at once
            loss += self.loss.compute(y[i], self.predict(sample, verbose=False))

        e = Evaluation(np.divide(loss, x.shape[0]), metrics=None)

        return [e.loss]
