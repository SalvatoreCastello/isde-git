import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        pass

    def predict(self, xts):
        n_samples = xts.shape[0]
        n_classes = self.centroids.shape[0]
        dist = np.zeros(shape=(n_samples, n_classes))
        for k in range(0, n_classes):
            dist[:, k] = np.sum((xts - self.centroids[k, :]) ** 2, axis=1)  # broadcasting
        ypred = np.argmin(dist, axis=1)
        return ypred

    def fit(self, xtr, ytr):

        if not isinstance(xtr, np.ndarray):
            raise TypeError("inputs should be ndarrays")

        if xtr.shape[0] != ytr.size:
            raise ValueError("input sizes are not consistent")

        n_classes = np.unique(ytr).size
        n_features = xtr.shape[1]
        self._centroids = np.zeros(shape=(n_classes, n_features))
        for k in range(0, n_classes):
            self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)
        return self