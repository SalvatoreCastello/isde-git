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
    """Split the data X,y into two random subsets.

    input:
        x: set of images
        y: labels
        fract_tr: float, percentage of samples to put in the training set.
            If necessary, number of samples in the training set is rounded to
            the lowest integer number.

    output:
        Xtr: set of images (numpy array, training set)
        Xts: set of images (numpy array, test set)
        ytr: labels (numpy array, training set)
        yts: labels (numpy array, test set)

    """
    n_samples = x.shape[0]
    idx = list(range(0, n_samples))  # [0 1 ... 999]  np.linspace
    np.random.shuffle(idx)
    n_tr = int(tr_fraction * n_samples)

    idx_tr = idx[:n_tr]
    idx_ts = idx[n_tr:]

    xtr = x[idx_tr, :]
    ytr = y[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]

    return xtr, ytr, xts, yts


def load_mnist(csv_filename):
    # loads data from a CSV file hosted in our repository
    data = pd.read_csv(csv_filename)
    data = np.array(data)  # cast pandas dataframe to numpy array

    y = data[:, 0]
    X = data[:, 1:]

    return X, y
