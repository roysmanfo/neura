from typing import Optional
import numpy as np


class Evaluation():

    def __init__(self, loss: np.float64, metrics: Optional[dict[str, np.float64]] = None) -> None:
        self.loss = loss
        self.metrics = metrics
