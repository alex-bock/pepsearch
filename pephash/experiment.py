
from functools import partial
from multiprocessing import Pool
from time import time
from tqdm import tqdm
from typing import Dict, List

import numpy as np
from torch import Tensor

from pephash.library import Library
from pephash.representations import Representation
from pephash.projection import Projection
from pephash.table import Table
from pephash.metrics import Metric


class Experiment:

    def __init__(self, model: Representation, projection: Projection):

        self.representation = model
        self.projection = projection

        self.table = Table(projection=projection)

        return

    def load_library(self, library: Library, X: Tensor):

        self.library = library

        for i in tqdm(range(len(X))):
            self.table.insert(x=X[i], key=i)

        return

    def quantify_table(self):

        return [
            len(bucket) if bucket is not None else 0 for bucket in self.table
        ]

    def evaluate(
        self, query_library: Library, X: Tensor, metric: Metric
    ) -> Dict:

        results = list()

        for i in tqdm(range(len(X))):

            t_0 = time()
            neighbors = self.table.query(X[i])
            t_i = time() - t_0

            if len(neighbors) == 0:
                avg_neighbor_distance = np.nan
                stdev_neighbor_distance = np.nan
                closest_neighbor_seq = ""
                closest_neighbor_distance = np.nan
                t_f = t_i
            else:
                neighbor_seqs = [self.library[j] for j in neighbors]
                neighbor_distances = [
                    metric(query_library[i], neighbor)
                    for neighbor in neighbor_seqs
                ]
                t_f = time() - t_0
                avg_neighbor_distance = np.mean(neighbor_distances)
                stdev_neighbor_distance = np.std(neighbor_distances)
                closest_neighbor_idx = np.argmin(neighbor_distances)
                closest_neighbor_seq = neighbor_seqs[closest_neighbor_idx]
                closest_neighbor_distance = neighbor_distances[
                    closest_neighbor_idx
                ]

            result = {
                "closest_neighbor_seq": closest_neighbor_seq,
                "closest_neighbor_distance": closest_neighbor_distance,
                "avg_neighbor_distance": avg_neighbor_distance,
                "stdev_neighbor_distance": stdev_neighbor_distance,
                "query_time": t_f
            }
            results.append(result)

        return results


def _get_ground_truth(
    i: int, query_library: Library, ref_library: Library, metric: Metric
) -> Dict:

    query_seq = query_library[i]

    t_0 = time()
    neighbor_distances = [
        metric(query_seq, ref_library[i])
        for i in tqdm(range(len(ref_library)))
    ]
    t_f = time() - t_0

    avg_neighbor_distance = np.mean(neighbor_distances)
    stdev_neighbor_distance = np.std(neighbor_distances)
    closest_neighbor_idx = np.argmin(neighbor_distances)
    closest_neighbor_seq = ref_library[closest_neighbor_idx]
    closest_neighbor_distance = neighbor_distances[closest_neighbor_idx]

    result = {
        "actual_neighbor_seq": closest_neighbor_seq,
        "actual_neighbor_distance": closest_neighbor_distance,
        "avg_distance": avg_neighbor_distance,
        "stdev_distance": stdev_neighbor_distance,
        "compute_time": t_f,
        "actual_distances": neighbor_distances
    }

    return result


def get_ground_truth(
    query_library: Library, ref_library: Library, metric: Metric
) -> List[Dict]:

    results = list()

    pool = Pool()
    results = pool.map(
        partial(
            _get_ground_truth, query_library=query_library,
            ref_library=ref_library, metric=metric
        ),
        range(len(query_library))
    )

    return results
