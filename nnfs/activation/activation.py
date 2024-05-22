from abc import ABC as _ABC, abstractmethod as _abstractmethod
import math as _math
import random as _random

class ActivationFunction(_ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @_abstractmethod
    def apply_formula(self, value: float) -> float:
        """
        Apply the formula of the activation function
        """
        ...
    
    @_abstractmethod
    def derivative(self, value: float) -> float:
        """
        Apply the formula of the activation function
        """
        ...

class Linear(ActivationFunction):
    """
    ## f(x) = x
    """
    def __init__(self) -> None:
        super().__init__("linear")

    def apply_formula(self, value: float) -> float:
        return value
    
    def derivative(self, value: float) -> float:
        return 1
        
class Sigmoid(ActivationFunction):
    """
    ## f(x) = 1 / (1 + e^(-x))
    """
    def __init__(self) -> None:
        super().__init__("sigmoid")

    def apply_formula(self, value: float) -> float:
        return 1 / (1 + _math.exp(-value))

    def derivative(self, value: float) -> float:
        return _math.exp(value) / ((1 + _math.exp(value)) ** 2)
    
class Exponential(ActivationFunction):
    """
    ## f(x) = e^(x)
    """
    def __init__(self) -> None:
        super().__init__("exponential")

    def apply_formula(self, value: float) -> float:
        return _math.exp(value)
    
    def derivative(self, value: float) -> float:
        return _math.exp(value)

class ReLu(ActivationFunction):
    """
    ## f(x) = max(0, x)
    """
    def __init__(self) -> None:
        super().__init__("relu")

    def apply_formula(self, value: float) -> float:
        return max(0, value)

    def derivative(self, value: float) -> float:
        return 0 if value <= 0 else 1
    
class LeakyReLu(ActivationFunction):
    """
    ## f(x) = max(x, 0.1x)
    """
    def __init__(self) -> None:
        super().__init__("leakyrelu")

    def apply_formula(self, value: float) -> float:
        return max(value, .1 * value)

    def derivative(self, value: float) -> float:
        return 1 if value >= 0 else .1

class Tanh(ActivationFunction):
    """
    ## f(x) = tanh(x)
    """
    def __init__(self) -> None:
        super().__init__("tanh")

    def apply_formula(self, value: float) -> float:
        return _math.tanh(value)

    def derivative(self, value: float) -> float:
        return _math.cosh(value) ** -2 
    
class Swish(ActivationFunction):
    """
    ## f(x) = x * sigmoid(x)
    """
    def __init__(self) -> None:
        super().__init__("swish")
        self.sigmoid = Sigmoid()
    
    def apply_formula(self, value: float) -> float:
        return value * self.sigmoid.apply_formula(value)

    def derivative(self, value: float) -> float:
        return self.sigmoid.apply_formula(value) + value * self.sigmoid.derivative(value)
    
class PReLU (ActivationFunction):
    """
    ## f(x) = ax if x < 0 else x
    """
    
    def __init__(self) -> None:
        super().__init__("prelu")
        self.a = _random.gauss()
    
    def apply_formula(self, value: float) -> float:
        if value < 0:
            return self.a * value
        return value

    def derivative(self, value: float) -> float:
        if value < 0:
            return self.a
        return 1    

    