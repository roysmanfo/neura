import numpy as np
import random as _random
from typing import Any, List, Optional, Union

from neura import optimizers
from neura import losses
from neura import preprocessing 
from neura.losses import Loss
from neura.layers import exceptional, Layer, Input
from neura.utils.types import InputValue, OutputValue
from neura.evaluation import Evaluation


class Model:
    def __init__(self, layers: Optional[list[Layer]] = None, name: Optional[str] = None) -> None:
        """
        parameters:
        
        - layers:         a list of layers to initialize the model. you can add more layers
                          by calling `add_layer()`
        - name:           the name of the model
                    
        """
        
        if not isinstance(name, str) and not name is None:
            raise ValueError("name must be of type str or None")
        
        self.loss: Loss = losses.MeanSquaredError()
        self.optimizer: optimizers.Optimizer = optimizers.SGD()
        self.metrics = []

        self.layers: list[Layer] = []
        self.name = "Model" if name is None else str(name) or "Model"
        
        self.output_shape: tuple[int, ...] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.input_size: int | None = None
        
        if layers:
            input_shape = layers[0].input_shape

            if input_shape is None:
                raise ValueError("no input_shape provided for the first layer")
       
            if input_shape and not all(isinstance(i, int) for i in input_shape):
                raise ValueError("input_shape can only contain int values")

            self.input_size = Input(input_shape) if input_shape else 1
            self.input_shape = input_shape

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
            
            while n < len(self.layers) and self.layers[-n].pass_trough_layer:
                n += 1

            if n == len(self.layers):
                # no upper layer was able to provide an actuall input_shape for this 
                # layer, use the network's input_shape then 
                previous_layer_node_count = self.input_size
            else:
                prev_layer = self.layers[-1]
                if prev_layer.output_shape and isinstance(prev_layer.output_shape, tuple):
                    previous_layer_node_count = Input(prev_layer.output_shape)
                else:
                    previous_layer_node_count = len(self.layers[-1].nodes)
                
                # assert that the provided input_shape is compatible with the rest of the network
                if layer.input_shape and layer.input_shape != prev_layer.output_shape:
                    raise RuntimeError(f"the input shape of layer {len(self.layers) + 1} " \
                                       f"({layer.name}) is incompatible with output shape "
                                       f"{prev_layer.output_shape} ({prev_layer.name})")
                
                layer.input_shape = prev_layer.output_shape

            for n in layer.nodes:
                n.weights = np.array([_random.uniform(-1, 1) for _ in range(previous_layer_node_count or 1)])
        else:
            if isinstance(layer, exceptional.NotFirstLayer):
                raise RuntimeError("%s (%s)" % (exceptional.NotFirstLayer.errmsg, layer.name))

            if not layer.input_shape:
                raise ValueError("no input_shape provided for the first layer")
            
            if layer.input_shape and not all(isinstance(i, int) for i in layer.input_shape):
                raise ValueError("input_shape can only contain int values")

            self.input_size = Input(layer.input_shape)
            for n in layer.nodes:
                n.weights = np.array([_random.uniform(-1, 1) for _ in range(self.input_size)])

        self.layers.append(layer)
        
        # modify the output shape
        if layer.output_shape:
            self.output_shape = layer.output_shape
        else:
            # in this case the output shape depends on the number of nodes
            self.output_shape = (len(layer.nodes),)        


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
            values = layer.forward(values)

        if verbose:
            print(f"predicting (layer: {len(self.layers)} / {len(self.layers)})")
        
        return values
    
    def forward(self, x: InputValue) -> OutputValue:
        return self.predict(x, verbose=False)

    def compute_loss(self, y_true: InputValue, y_pred: InputValue):
        return self.loss(y_true, y_pred)

    def backward(self, y_true: InputValue, y_pred: InputValue) -> None:
        output_gradient = self.loss.derivative(y_true, y_pred)
        
        # Backpropagate through the layers
        for layer in reversed(self.layers):
            if layer.trainable:
                gradients = layer.compute_gradients(output_gradient)
                layer.update_weights(self.optimizer, gradients)
                
                # this will be needed by the previous layer
                output_gradient = np.sum(gradients, axis=0)

    def train(self,
        x: InputValue,
        y: InputValue,
        batch_size: int = 16,
        epochs: int = 5,
        shuffle: bool = False,
        verbose: bool = True) -> None:

        if len(x) != len(y):
            raise ValueError("X and y are not the same size (len(x) != len(y))")

        elif not isinstance(epochs, int) and epochs < 1:
            raise ValueError("epochs must be an int >=1")

        elif not isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("batch_size must be an int >=1")
        
        def set_training_flag(value: bool):            
            def set_flag(l: Layer): l.training = value
            return set_flag

        map(set_training_flag(True), self.layers)

        for epoch in range(epochs):
            if shuffle and self._is_batch_input(x):
                x, y = preprocessing.shuffle(x, y)

            # x_batches = [x[i:i + batch_size] for i in range(0, len(x), batch_size)]
            # y_batches = [y[i:i + batch_size] for i in range(0, len(y), batch_size)]

            epoch_loss = 0

            for i, data in enumerate(x):
                y_pred = self.forward(data)
                epoch_loss += self.compute_loss(y[i], y_pred)
                self.backward(y[i], y_pred)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(x)}")
        
        map(set_training_flag(False), self.layers)

    def _is_batch_input(self, x: InputValue) -> bool:
        if not self.layers:
            raise RuntimeError("no layers have been added yet")

        input_shape = self.layers[0].input_shape
        if not input_shape:
            raise RuntimeError("no input_shape has been provided")

        if len(input_shape) + 1 != len(x.shape):
            return False

        return True


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
        
        if not self._is_batch_input(x):
            input_shape = self.layers[0].input_shape
            if not input_shape:
                raise ValueError("no input_shape has been provided") # just do it, ... please 
            
            raise ValueError("unable to process batch input_shape {}. " \
                             "expected shape: {}".format(x.shape, '(n, ' + ', '.join(str(i) for i in input_shape) + ')'))

        
        loss = 0
        for i, sample in enumerate(x):
            # TODO: reduce overhead and increase efficiency by processing all the data at once
            loss += self.compute_loss(y[i], self.predict(sample, verbose=False))

        e = Evaluation(np.divide(loss, x.shape[0]), metrics=None)

        return [e.loss]
