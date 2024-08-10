from typing import Any
import numpy as np

Float64 = np.float64
DtFloat64 = np.dtype[np.float64]

InputValue = np.ndarray[Any, DtFloat64]
OutputValue = np.ndarray[Any, DtFloat64]

NodeWeight = np.dtype[np.float32]
NodeWeights = np.ndarray[Any, NodeWeight]
NodeOutput = Float64

Gradient = DtFloat64
Gradients = np.ndarray[Any, Gradient]
