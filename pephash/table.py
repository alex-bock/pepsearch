
from bitarray import bitarray
from typing import Hashable, List, Union

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
