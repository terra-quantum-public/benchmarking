import numpy as np
import pandas as pd
from scipy.special import erfc
import timing
import make_data



def chauvenet_stats(array: np.ndarray):
    """
    Calculate statistics of an array after rejecting outliers with Chauvenet's criterion.

    Args
    ____
    array : np.ndarrray
         Input array of values.

    Returns
    _______
    pd.DataFrame

    """
    mean: float = np.mean(array)  # Mean of incoming array
    stdv: float = np.std(array)  # Standard deviation
    N: int = len(array)  # Length of incoming array
    criterion: float = 1.0 / (2 * N)  # Chauvenet's criterion
    d: np.ndarray = abs(array - mean) / stdv  # Distance of a value to mean in stdv's
    prob: np.ndarray = erfc(d)  # Area normal dist.
    bool_values: np.ndarray = prob > criterion  # Use boolean array outside this function
    return np.mean(array[bool_values]), np.std(array[bool_values])


def make_stats_table(stats: np.ndarray , models: dict, N: int):
    """
    Creates DataFrame with model stats

    Args
    ____
    stats : np.ndarray(len(models), N//2, 2)
        Final entry of stats should contain mean then std
    models : dict
        dict with keys of str names and values of nn.Modules
    N : int
        largest_qubit_circuit_size

    Returns
    _______
    pd.DataFrame
    """
    columns = []
    for n in range(N // 2):
        columns.append(f"{2 * (n + 1)} Qubits Mean")
        columns.append(f"{2 * (n + 1)} Qubits STD")

    stats_df = np.zeros((len(models), N // 2 * 2))

    stats_df[:, 0::2] = stats[:, :, 0]
    stats_df[:, 1::2] = stats[:, :, 1]

    return pd.DataFrame(stats_df, columns=columns, index=models.keys())
