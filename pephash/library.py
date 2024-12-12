
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from Bio.Align.substitution_matrices import Array as SubMatrix

from pephash.constants import ALPHABET, BLOSUM62
from pephash.representations import Representation
from pephash.utils import convert_logs, reduce_alphabet


tqdm.pandas()


class Library:

    def __init__(
        self, n: int, m: int = None, seed_seq: str = None,
        sub_matrix: SubMatrix = None
    ):

        self.n = n

        if seed_seq is not None:
            if sub_matrix is None:
                sub_matrix = convert_logs(reduce_alphabet(BLOSUM62, alphabet=ALPHABET))
            seqs = self._from_seed_seq(seed_seq, sub_matrix=sub_matrix)
            self.m = len(seed_seq)
        else:
            self.m = m
            seqs = list()
            for _ in range(self.n):
                seq = "".join(np.random.choice(list(ALPHABET), size=self.m))
                seqs.append(seq)

        self.seqs = pd.Series(seqs)
        self.motif = self._compute_motif()

        return

    def _from_seed_seq(self, seed_seq: str, sub_matrix: SubMatrix) -> List[str]:

        seqs = list()
        for _ in tqdm(range(self.n)):
            seq = "".join(
                [np.random.choice(
                    list(ALPHABET),
                    p=np.array([sub_matrix[c][token] for token in list(ALPHABET)]) / np.sum([sub_matrix[c][token] for token in list(ALPHABET)]))
                for c in seed_seq]
            )
            seqs.append(seq)

        return seqs

    def _compute_motif(self) -> Dict:

        motif = {key: list() for key in list(ALPHABET)}
        for i in range(self.m):
            counts = self.seqs.str[i].value_counts()
            for token in ALPHABET:
                motif[token].append(counts.get(token, 0) / counts.sum())

        return

    def __len__(self) -> int:

        return len(self.seqs)

    def __getitem__(self, idx: int) -> str:

        return self.seqs.iloc[idx]

    def P(self, seq: str) -> float:

        if len(seq) != self.m:
            raise ValueError

        p = 1.0
        for i in range(self.m):
            p *= self.motif[seq[i]][i]

        return p

    def generate(self) -> str:

        x = str()
        for i in range(self.m):
            x += np.random.choice(
                list(ALPHABET), p=[self.motif[token][i] for token in ALPHABET]
            )

        return x

    def represent(self, model: Representation) -> Tensor:

        return torch.stack(self.seqs.progress_apply(model).values.tolist())
