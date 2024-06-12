from abc import ABC as _ABC, abstractmethod as _abstractmethod
from typing import Optional


class BaseOptimizer(_ABC):
    def __init__( self, learning_rate: float, name: Optional[str] = None ):

        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.learning_rate = learning_rate
        
    @_abstractmethod
    def apply_gradients() -> None:
        raise NotImplementedError("This method should be implemented in a subclass")



