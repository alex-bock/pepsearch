
import abc
from typing import List, Union

import torch
from torch import nn, Tensor

import esm

from .constants import ATCHLEY_FACTORS, KIDERA_FACTORS, BLOSUM_INDICES


class Representation(nn.Module):

    def __init__(self):

        super(Representation, self).__init__()

        return

    @abc.abstractmethod
    def forward(self, input: str) -> Tensor:

        raise NotImplementedError


class Atchley(Representation):

    dim = 5

    def __init__(self, flatten: bool = False):

        super(Atchley, self).__init__()

        self.flatten = flatten

        return

    def forward(self, input: str) -> Tensor:

        X = torch.stack([Tensor(ATCHLEY_FACTORS[c]) for c in input])
        if self.flatten:
            X = X.flatten()

        return X


class Kidera(Representation):

    dim = 10

    def __init__(self, flatten: bool = False):

        super(Kidera, self).__init__()

        self.flatten = flatten

        return

    def forward(self, input: str) -> Tensor:

        X = torch.stack([Tensor(KIDERA_FACTORS[c]) for c in input])
        if self.flatten:
            X = X.flatten()

        return X


class BLOSUMIndices(Representation):

    dim = 10

    def __init__(self, flatten: bool = False):

        super(BLOSUMIndices, self).__init__()

        self.flatten = flatten

        return

    def forward(self, input: str) -> Tensor:

        X = torch.stack([Tensor(BLOSUM_INDICES[c]) for c in input])
        if self.flatten:
            X = X.flatten()

        return X


class ESMEmbedding(Representation):

    dim = 320

    def __init__(self):

        super(ESMEmbedding, self).__init__()

        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        return

    def forward(self, input: Union[str, List[str]]) -> Tensor:

        if isinstance(input, str):
            input = [input]

        data = [(i, seq) for i, seq in enumerate(input)]
        _, _, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(
                batch_tokens, repr_layers=[6], return_contacts=True
            )

        token_representations = results["representations"][6]
        X = token_representations.mean(axis=1)

        if len(X) == 1:
            X = X[0]

        return X
