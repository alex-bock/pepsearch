
from bitarray import bitarray
from typing import Hashable, List, Union

from sklearn.utils import murmurhash3_32
from torch import Tensor

from .projection import Projection


class Table:

    def __init__(self, projection: Projection):

        self.table = [None for _ in range(2 ** projection.k)]
        self.projection = projection

        return

    def __getitem__(self, idx: int) -> Union[List, None]:

        return self.table[idx]

    def hash(self, x: Tensor):

        x = self.projection(x)
        if len(x.shape) == 2 and x.shape[0] == 1:
            x = x[0]

        return bitarray(list(x > 0)).to01()

    def insert(self, x: Tensor, key: Hashable):

        index = int(self.hash(x), base=2)
        if self.table[index] is None:
            self.table[index] = [key]
        else:
            self.table[index].append(key)

        return

    def query(self, x: Tensor):

        index = int(self.hash(x), base=2)
        neighbors = set()
        if self.table[index] is not None:
            neighbors = neighbors.union(self.table[index])

        return neighbors


class MurmurHashTable:

    def __init__(self, k: int):

        self.k = 2 ** k
        self.table = [None for _ in range(self.k)]

        return

    def __getitem__(self, idx: int) -> Union[List, None]:

        return self.table[idx]

    def hash(self, x: str):

        return murmurhash3_32(x) % self.k

    def insert(self, x: str, key: Hashable):

        index = self.hash(x)
        if self.table[index] is None:
            self.table[index] = [key]
        else:
            self.table[index].append(key)

        return

    def query(self, x: str):

        index = self.hash(x)
        neighbors = set()
        if self.table[index] is not None:
            neighbors = neighbors.union(self.table[index])

        return neighbors
