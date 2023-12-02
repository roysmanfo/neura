from abc import ABC as _ABC, abstractmethod as _abstractmethod
import math as _math

class ActivationFunction(_ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @_abstractmethod
    def apply_formula(self, value: float) -> float:
        """
        Apply the formula of the activation function
        """
        ...

class Basic(ActivationFunction):
    """
    ## f(x) = x
    """
    def __init__(self) -> None:
        super().__init__("basic")

    def apply_formula(self, value: float) -> float:
        return value
    
class Sigmoid(ActivationFunction):
    """
    ## f(x) = 1 / (1 + e^(-value))
    """
    def __init__(self) -> None:
        super().__init__("sigmoid")

    def apply_formula(self, value: float) -> float:
        return 1 / (1 + _math.exp(-value))
    
class ReLu(ActivationFunction):
    """
    ## f(x) = max(0, x)
    """
    def __init__(self) -> None:
        super().__init__("relu")

    def apply_formula(self, value: float) -> float:
        return max(0, value)
    
class LeakyReLu(ActivationFunction):
    """
    ## f(x) = max(x, 0.1x)
    """
    def __init__(self) -> None:
        super().__init__("leakyrelu")

    def apply_formula(self, value: float) -> float:
        return max(value, .1 * value)

class Tanh(ActivationFunction):
    """
    ## f(x) = tanh(x)
    """
    def __init__(self) -> None:
        super().__init__("tanh")

    def apply_formula(self, value: float) -> float:
        return _math.tanh(value)

class Derivative(ActivationFunction):
    """
    ## f(x) = 1 if x > 0 else 0
    """
    def __init__(self) -> None:
        super().__init__("derivative")

    def apply_formula(self, value: float) -> float:
        return 1 if value > 0 else 0
    
    