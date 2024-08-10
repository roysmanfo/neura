from typing import Any

from neura.optimizers import Optimizer
from neura.utils.types import Gradients, NodeWeights 

class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    ===========================

    Use the `Gradient Descent` algorithm to minimize the loss function's output.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 name: str = "SGD",
                 **kwargs: Any
                 ) -> None:
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)

    def apply_gradients(self, weights: NodeWeights, gradients: Gradients) -> NodeWeights:
        # Update the weights in place and return a reference
        weights -= self.learning_rate * gradients
        return weights
