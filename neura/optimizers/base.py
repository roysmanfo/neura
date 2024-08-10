from abc import ABC as _ABC, abstractmethod as _abstractmethod
from typing import Optional

from neura.utils.types import NodeWeights, Gradients


class Optimizer(_ABC):
    def __init__( self, learning_rate: float, name: Optional[str] = None ):

        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.learning_rate = learning_rate


    @_abstractmethod
    def apply_gradients(self, weights: NodeWeights, gradients: Gradients) -> NodeWeights:
        """
        Update the weights of a node in place using the calculated gradients.

        Parameters
        ----------
            weights: np.ndarray
                Current weights of the node
            gradients: np.ndarray
                Gradients computed for the node with respect to the loss

        Returns
        -------
            out: np.ndarray
                Updated weights after applying SGD
        """
        raise NotImplementedError("This method should be implemented in a subclass")



