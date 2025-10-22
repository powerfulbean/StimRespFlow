import numbers
from typing import TypeVar

try:
    import torch
except:
    torch = None

try:
    import numpy as np
except:
    np = None

Array = TypeVar("Array")

def get_numeric_backend(data: Array):
    if isinstance(data, torch.Tensor):
        return torch
    elif isinstance(data, np.ndarray):
        return np
    elif isinstance(data, numbers.Number):
        return np
    else:
        raise ValueError(f"input is not an numeric variable")

def is_tensor(data: Array):
    if isinstance(data, torch.Tensor):
        return True
    elif isinstance(data, np.ndarray):
        return False
    elif isinstance(data, numbers.Number):
        return False
    else:
        raise ValueError(f"input is not an numeric variable")