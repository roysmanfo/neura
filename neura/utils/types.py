from typing import Any
import numpy as np

InputValue = np.ndarray[Any, np.dtype[np.float64]]
OutputValue = np.ndarray[Any, np.dtype[np.float64]]

NodeWeight = np.dtype[np.float32]
NodeWeights = np.ndarray[Any, NodeWeight]

Gradient = np.dtype[np.float64]
Gradients = np.ndarray[Any, Gradient]
