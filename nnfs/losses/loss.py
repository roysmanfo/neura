import numpy as np
from abc import ABC, abstractmethod

from nnfs.utils.types import InputValue



class Loss(ABC):
    """
    Base class for all loss functions

    All classes that inherit this class have to redefine the method `compute()`
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def compute(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        """
        Compute the cost of the batch

        Parameters:
            :param y_true (np.ndarray): The true values.
            :param y_pred (np.ndarray): The predicted values.

        Returns:
            np.float64: The computed cost of the batch (loss).
        """
        raise NotImplementedError("This method needs to be implement in a subclass")


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE)

    more robust to outliers than MSE
    """

    def compute(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        return np.mean(np.abs(y_true - y_pred))



class MeanSquaredError(Loss):
    """
    Mean Squared Error (MAE)

    Most commonly used loss function
    - penalizes the model by producing large errors even for small mistakes 
    """

    def compute(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        return np.mean((y_true - y_pred) ** 2)
    
class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss function (Log Loss)
    
    This is mostly used for categorical models
    - works well in the interval [0, 1] (2 classes)
    """
    
    def compute(self, y_true: InputValue, y_pred:InputValue) -> np.float64:
        # ? avoid calculating log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) 

class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss function
    
    This is mostly used for categorical models
    - works like BCE, but supports multiple classes 
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        # ? avoid calculating log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred))

class HingeLoss(Loss):
    """
    Hinge Loss

    This is mostly used for categorical models
    - uses to penalize both wrong and insecure answers
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:        
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

class LogCoshLoss(Loss):
    """
    Log-Cosh Loss

    - This loss function is not so affected by occasional mistakes like MSE
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:        
        return np.mean(np.log(np.cosh(y_pred - y_true)))

class HuberLoss(Loss):
    """
    Huber Loss

    This loss function is less sensitive to outliers in data than the squared error loss.    
    - It is quadratic for small errors and linear for large errors.
      (combination of MSE and MAE)
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initializes the Huber loss with a given delta value.
        
        Parameters:
            :param delta (float): The threshold at which to switch from quadratic to linear loss.
        """
        self.delta = delta

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    