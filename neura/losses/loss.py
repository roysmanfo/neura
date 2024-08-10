from typing import Any
import numpy as np
from abc import ABC, abstractmethod
from neura.utils.types import InputValue, OutputValue


class Loss(ABC):
    """
    Base class for all loss functions

    All classes that inherit this class have to redefine the methods `compute()` and `derivative()`
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        """
        Compute the cost of the batch

        Parameters
        ----------
        y_true: np.ndarray
            The true values.
        y_pred: np.ndarray
            The predicted values.

        Returns
        -------
        out: np.float64
            The computed cost of the batch (loss).
        """
        raise NotImplementedError(
            "This method needs to be implement in a subclass")

    @abstractmethod
    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        """
        Compute the derivative

        Parameters
        ----------
        y_true: np.ndarray
            The true values.
        y_pred: np.ndarray
            The predicted values.

        Returns
        -------
        out: np.ndarray
            The derivative computed in y_pred.
        """
        raise NotImplementedError(
            "This method needs to be implement in a subclass")


class ParametricLoss(Loss):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE)
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        return np.where(y_pred > y_true, 1, -1) / y_true.size


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE)
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        return (2 * (y_pred - y_true)) / y_true.size


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss function (Log Loss)
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        # ? avoid calculating log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss function
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        # ? avoid calculating log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred))

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> InputValue:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) / y_true.shape[0]


class HingeLoss(Loss):
    """
    Hinge Loss
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        return np.where(y_true * y_pred < 1, -y_true, 0) / y_true.size


class LogCoshLoss(Loss):
    """
    Log-Cosh Loss
    """

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        return np.mean(np.log(np.cosh(y_pred - y_true)))

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        return np.tanh(y_pred - y_true) / y_true.size


class HuberLoss(ParametricLoss):
    """
    Huber Loss
    """

    def __init__(self, delta: float = 1.0):
        """
        Initializes the Huber loss with a given delta value.

        Parameters
        ----------
        delta: float
            The threshold at which to switch from quadratic to linear loss.
        """
        self.delta = delta

    def compute(self, y_true: InputValue, y_pred: InputValue) -> np.float64:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * np.square(error)
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)

        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def derivative(self, y_true: InputValue, y_pred: InputValue) -> OutputValue:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        return np.where(is_small_error, error, self.delta * np.sign(error)) / y_true.size
