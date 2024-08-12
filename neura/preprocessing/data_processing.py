"""
All data preprocessing related functions and methods
"""


import numpy as np
from neura.utils.types import InputValue, OutputValue



def shuffle(*arrays: InputValue) -> tuple[OutputValue, ...]:
    """
    Shuffle 2 or more arrays, while preserving order
    
    Note: all the arrays must have the same lenght

    Parameters
    ----------
    arrays : np.ndarray
        Input arrays to be split. All arrays must have the same length.

    eturns
    -------
    out: tuple[np.ndarray, ...]
        the shuffled arrays
    """

    perm = np.random.permutation(len(arrays[0]))
    return tuple(array[perm] for array in arrays)

def validation_split(*arrays: InputValue, val_split: float = .25) -> tuple[OutputValue, ...]:
    """
    split both X and y in 2 while maintaing order
    
    In case more than 2 arrays have been provided,
    those will be splitted the same way

    Note: all the arrays must have the same lenght

    Parameters
    ----------
    arrays : np.ndarray
        Input arrays to be split. All arrays must have the same length.
    val_split : float
        Fraction of the data to use for the validation set.

    Returns
    -------
    out: tuple[np.ndarray, ...]
        X_train, y_train, X_test, y_test , ...

    """

    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    
    if len(set([len(arr) for arr in arrays])) != 1:
        raise ValueError("All input arrays must have the same length.")
    
    split_index = int(len(arrays[0]) * (1 - val_split))
    
    train_arrays = [arr[:split_index] for arr in arrays]
    val_arrays = [arr[split_index:] for arr in arrays]

    return (*train_arrays, *val_arrays)
    



    




