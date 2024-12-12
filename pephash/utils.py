
import numpy as np

from Bio.Align.substitution_matrices import Array as SubMatrix


def reduce_alphabet(sub_matrix: SubMatrix, alphabet: str) -> SubMatrix:

    data = [
        [sub_matrix[t1][t2] for t2 in list(alphabet)] for t1 in list(alphabet)
    ]

    return SubMatrix(alphabet=alphabet, data=np.array(data))


def convert_logs(
    sub_matrix: SubMatrix, base: float = 2.0, reverse: bool = False
) -> SubMatrix:

    data = base ** (np.array(sub_matrix.data) * ((-1) ** reverse))

    return SubMatrix(alphabet=sub_matrix.alphabet, data=data)


def normalize(sub_matrix: SubMatrix) -> SubMatrix:

    data = [
        sub_matrix[token] / sub_matrix[token].sum()
        for token in sub_matrix.alphabet
    ]

    return SubMatrix(alphabet=sub_matrix.alphabet, data=np.array(data))
