from typing import Any
import numpy as np

Float64 = np.float64
DtFloat64 = np.dtype[np.float64]

InputValue = np.ndarray[Any, DtFloat64]
OutputValue = np.ndarray[Any, DtFloat64]

NodeWeight = np.float64
NodeWeights = np.ndarray[Any, np.dtype[np.float64]]
NodeOutput = Float64

Gradient = DtFloat64
Gradients = np.ndarray[Any, Gradient]
