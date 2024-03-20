from typing import List
from .loss import Loss


class MeanAbsoluteError(Loss):
    """
    L(a) = |a|
    """

    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        s = sum([a - b for a, b in zip(y_true, y_pred)])
        return abs(s)


class MeanSquaredError(Loss):
    """
    L(a) = a^2
    """

    def call(self, y_true: List[float], y_pred: List[float]) -> float:
        s = sum([(a - b)**2 for a, b in zip(y_true, y_pred)])
        return s / len(y_true)
