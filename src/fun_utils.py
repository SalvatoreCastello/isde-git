import pandas as pd
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = pd.read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass

def load_mnist(csv_filename):
    # loads data from a CSV file hosted in our repository
    data = pd.read_csv(csv_filename)
    data = np.array(data)  # cast pandas dataframe to numpy array

    y = data[:, 0]
    X = data[:, 1:]

    return X, y