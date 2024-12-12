
import abc
from typing import Union

import numpy as np
import plotly.express as px
from scipy.spatial import distance_matrix
from scipy.spatial.distance import hamming, euclidean, cosine
from scipy.stats import spearmanr
from torch import nn, Tensor

from Bio.Align.substitution_matrices import Array

from .constants import BLOSUM62, ALPHABET
from .constants import ATCHLEY_FACTORS, KIDERA_FACTORS, BLOSUM_INDICES
from .utils import convert_logs


class Metric(nn.Module):

    def __init__(self):

        super(Metric, self).__init__()

        return

    @abc.abstractmethod
    def forward(self, input_1: str, input_2: str) -> float:

        raise NotImplementedError


class Tokenizer(nn.Module):

    def __init__(self, alphabet: str = ALPHABET):

        super(Tokenizer, self).__init__()

        self.alphabet = alphabet

        return

    def forward(self, input: str) -> Tensor:

        return Tensor([self.alphabet.find(c) for c in input])


class HammingDistance(Metric):

    def __init__(self):

        super(HammingDistance, self).__init__()

        self.tokenizer = Tokenizer()

        return

    def forward(
        self, input_1: Union[str, Tensor], input_2: Union[str, Tensor]
    ) -> Tensor:

        if isinstance(input_1, str):
            input_1 = self.tokenizer(input_1)
        if isinstance(input_2, str):
            input_2 = self.tokenizer(input_2)

        return hamming(input_1, input_2)


class FactorDistance(Metric):

    def __init__(self):

        super(FactorDistance, self).__init__()

        return

    def forward(self, input_1: str, input_2: str) -> float:

        n = len(input_1)
        if len(input_2) != n:
            raise ValueError

        return cosine(np.array([self.factors[input_1[i]] for i in range(n)]).flatten(), np.array([self.factors[input_2[i]] for i in range(n)]).flatten())

        return sum([np.linalg.norm(np.array(self.factors[input_1[i]]) - np.array(self.factors[input_2[i]])) for i in range(n)])

        return np.linalg.norm(
            [np.array(self.factors[input_1[i]]) - np.array(self.factors[input_2[i]]) for i in range(n)]
        )


class BLOSUMIndexDistance(FactorDistance):

    def __init__(self, *args, **kwargs):

        super(BLOSUMIndexDistance, self).__init__(*args, **kwargs)

        self.factors = BLOSUM_INDICES

        return


class SubstitutionScore(Metric):

    def __init__(self, normalize: bool = True):

        super(SubstitutionScore, self).__init__()

        self.normalize = normalize

        return

    def forward(self, input_1: str, input_2: str) -> float:

        n = len(input_1)
        if len(input_2) != n:
            raise ValueError

        factor = 1
        if self.normalize:
            factor /= n

        return sum(
            [int(input_1[i] != input_2[i]) * self.substitution_matrix[input_1[i], input_2[i]] for i in range(n)]
        ) * factor


class BLOSUM62Score(SubstitutionScore):

    def __init__(self, *args, **kwargs):

        super(BLOSUM62Score, self).__init__(*args, **kwargs)

        self.substitution_matrix = convert_logs(BLOSUM62, reverse=True)

        return


class AtchleyScore(SubstitutionScore):

    def __init__(self, *args, **kwargs):

        super(AtchleyScore, self).__init__(*args, **kwargs)

        keys, vectors = zip(*[
            (k, ATCHLEY_FACTORS.get(k, np.zeros(5))) for k in list(ALPHABET)
        ])
        vectors = np.array(vectors)
        dist_mat = distance_matrix(vectors, vectors)
        self.substitution_matrix = Array(alphabet="".join(keys), data=dist_mat)

        return


class KideraScore(SubstitutionScore):

    def __init__(self, *args, **kwargs):

        super(KideraScore, self).__init__(*args, **kwargs)

        keys, vectors = zip(*[
            (k, KIDERA_FACTORS.get(k, np.zeros(10))) for k in list(ALPHABET)
        ])
        vectors = np.array(vectors)
        dist_mat = distance_matrix(vectors, vectors)
        self.substitution_matrix = Array(alphabet="".join(keys), data=dist_mat)

        return


def compare_metrics(metric_1: Metric, metric_2: Metric):

    matrices = list()
    for metric in [metric_1, metric_2]:
        if isinstance(metric, SubstitutionScore):
            matrix = np.array(metric.substitution_matrix.data)
        elif isinstance(metric, FactorDistance):
            factors = np.array([v for v in metric.factors.values()])
            matrix = distance_matrix(factors, factors)
        matrices.append(matrix)

    px.imshow(matrices[0]).show()
    px.imshow(matrices[1]).show()

    print(spearmanr(matrices[0].flatten(), matrices[1].flatten()).statistic)

    return
