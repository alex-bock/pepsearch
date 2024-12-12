
from torch import nn, Tensor

import numpy as np
from sklearn.random_projection import GaussianRandomProjection as _GaussianRandomProjection


class Projection(nn.Module):

    pass


class RandomProjection(Projection):

    def __init__(self, d: int, k: int):

        super(RandomProjection, self).__init__()

        self.R = None

        self.k = k
        self.d = d

        return

    def fit(self, X: Tensor):

        self.R = np.random.rand(self.k, self.d) - 0.5

        return

    def forward(self, X: Tensor) -> Tensor:
        """
        N x d -> N x k
        """

        # handle singleton case
        if len(X.shape) == 1 and X.shape[0] == self.d:
            X = X.reshape(1, -1)

        return np.dot(X, self.R.T)


class GaussianRandomProjection(Projection):

    def __init__(self, d: int, k: int):

        super(GaussianRandomProjection, self).__init__()

        self.model = _GaussianRandomProjection(n_components=k)

        self.k = k
        self.d = d

        return

    def fit(self, X: Tensor):

        self.model.fit(X)

        return

    def forward(self, X: Tensor) -> Tensor:

        # handle singleton case
        if len(X.shape) == 1 and X.shape[0] == self.d:
            X = X.reshape(1, -1)

        return Tensor(self.model.transform(X))
