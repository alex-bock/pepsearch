
import json
import os
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pephash.library import Library
from pephash.representations import BLOSUMIndices
from pephash.projection import GaussianRandomProjection
from pephash.metrics import BLOSUMIndexDistance
from pephash.experiment import Experiment, get_ground_truth


if __name__ == "__main__":

    np.random.seed(1047)

    n = 1000000
    m = 50

    library = Library(n=n, m=m)
    query_set = Library(n=10, m=m)

    ks = [8, 9, 10, 11, 12, 13, 14]
    representation = BLOSUMIndices(flatten=True)
    metric = BLOSUMIndexDistance()
    projector = GaussianRandomProjection

    X = library.represent(representation)
    query_X = query_set.represent(representation)

    if os.path.exists("gt.json"):
        with open("gt.json", "r") as f:
            ground_truth = json.load(f)
        ground_truth_df = pd.read_csv("gt.csv")
    else:
        ground_truth = get_ground_truth(query_set, library, metric)
        with open("gt.json", "w") as f:
            json.dump(ground_truth, f)
        ground_truth_df = pd.DataFrame(ground_truth)
        ground_truth_df.to_csv("gt.csv")

    summary = dict()
    bucket_counts = dict()

    for k in ks:

        projection = projector(d=m * representation.dim, k=k)
        projection.fit(X)

        pprint(
            {
                "representation": representation,
                "metric": metric,
                "k": k,
                "projection": projection
            }
        )

        experiment = Experiment(
            model=representation, projection=projection
        )

        experiment.load_library(library, X)
        bucket_counts[k] = experiment.quantify_table()

        results = experiment.evaluate(query_set, query_X, metric)
        for i, result in enumerate(results):
            actual_distances = ground_truth[i]["actual_distances"]
            result["percentile"] = (
                np.array(actual_distances) < result["closest_neighbor_distance"]
            ).sum() / len(actual_distances)
            result.update(ground_truth[i])
            result.pop("actual_distances")
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"k={k}.csv")

        summary[k] = dict()
        for key in [
            "closest_neighbor_distance",
            "percentile",
            "avg_neighbor_distance",
            "stdev_neighbor_distance",
            "query_time",
            "actual_neighbor_distance",
            "avg_distance",
            "stdev_distance",
            "compute_time"
        ]:
            summary[k][key] = results_df[key].dropna(
                inplace=False
            ).values.mean()

    df = pd.DataFrame.from_dict(summary, orient="index")
    print(df)
    df.to_csv("summary.csv")

    fig = go.Figure()
    for k in ks:
        fig.add_trace(go.Histogram(x=bucket_counts[k], name=k))
    fig.update_layout(
        title=dict(text="Bucket size distribution"),
        xaxis_title=dict(text="# sequences"),
        yaxis_title=dict(text="# buckets"),
        barmode="stack"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index + 8, y=df["closest_neighbor_distance"], mode="lines",
            name="d(p, q')"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index + 8, y=df["actual_neighbor_distance"], mode="lines",
            name="d(p, q)"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index + 8, y=df["avg_distance"], mode="lines",
            name="Avg. d(p, x), x ∈ L"
        )
    )
    fig.update_layout(
        xaxis_title=dict(text="k"), yaxis_title=dict(text="Similarity"),
        yaxis_range=[0, 1]
    )
    fig.show()
