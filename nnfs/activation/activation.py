from abc import ABC as _ABC, abstractmethod as _abstractmethod
import math as _math
import random as _random

class ActivationFunction(_ABC):
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @_abstractmethod
    def apply_formula(self, x: float) -> float:
        """
        Apply the formula of the activation function
        """
        ...
    
    @_abstractmethod
    def derivative(self, x: float) -> float:
        """
        Calculate the derivative in x
        """
        ...

class Linear(ActivationFunction):
    """
    ## f(x) = x
    """

    def apply_formula(self, x: float) -> float:
        return x
    
    def derivative(self, x: float) -> float:
        return 1
        
class Sigmoid(ActivationFunction):
    """
    ## f(x) = 1 / (1 + e^(-x))
    """

    def apply_formula(self, x: float) -> float:
        return 1 / (1 + _math.exp(-x))

    def derivative(self, x: float) -> float:
        return _math.exp(x) / ((1 + _math.exp(x)) ** 2)
    
class Exponential(ActivationFunction):
    """
    ## f(x) = e^(x)
    """

    def apply_formula(self, x: float) -> float:
        return _math.exp(x)
    
    def derivative(self, x: float) -> float:
        return _math.exp(x)

class ReLu(ActivationFunction):
    """
    ## f(x) = max(0, x)
    """

    def apply_formula(self, x: float) -> float:
        return max(0, x)

    def derivative(self, x: float) -> float:
        return 0 if x <= 0 else 1
    
class LeakyReLu(ActivationFunction):
    """
    ## f(x) = max(x, 0.1x)
    """

    def apply_formula(self, x: float) -> float:
        return max(x, .1 * x)

    def derivative(self, x: float) -> float:
        return 1 if x >= 0 else .1

class Tanh(ActivationFunction):
    """
    ## f(x) = tanh(x)
    """

    def apply_formula(self, x: float) -> float:
        return _math.tanh(x)

    def derivative(self, x: float) -> float:
        return _math.cosh(x) ** -2 
    
class Swish(ActivationFunction):
    """
    ## f(x) = x * sigmoid(x)
    """
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = Sigmoid()
    
    def apply_formula(self, x: float) -> float:
        return x * self.sigmoid.apply_formula(x)

    def derivative(self, x: float) -> float:
        return self.sigmoid.apply_formula(x) + x * self.sigmoid.derivative(x)
    
class PReLU (ActivationFunction):
    """
    ## f(x) = ax if x < 0 else x
    """
    def __init__(self) -> None:
        super().__init__()
        self.a = _random.gauss()
    
    def apply_formula(self, x: float) -> float:
        if x < 0:
            return self.a * x
        return x

    def derivative(self, x: float) -> float:
        if x < 0:
            return self.a
        return 1    

    