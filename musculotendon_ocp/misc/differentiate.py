import numpy as np


def compute_finitediff(array: np.ndarray, t: np.ndarray) -> np.ndarray:
    finitediff = np.ndarray(len(t)) * np.nan
    finitediff[1:-1] = (array[2:] - array[:-2]) / (t[2] - t[0])
    return finitediff
