
import json
import os
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pephash.library import Library
from pephash.representations import BLOSUMIndices, ESMEmbedding
from pephash.projection import RandomProjection, GaussianRandomProjection
from pephash.metrics import compare_metrics, BLOSUM62Score, BLOSUMIndexDistance
from pephash.experiment import Experiment, get_ground_truth


if __name__ == "__main__":

    np.random.seed(1047)

    n = 1000000
    m = 50

    # library = Library(n=n, m=m)
    query_set = Library(n=10, m=m)
    print(query_set.seqs)
    exit()

    ks = [8, 9, 10, 11, 12, 13, 14]
    baseline_metric = BLOSUM62Score()
    other_metrics = [BLOSUMIndexDistance()]
    metrics = [baseline_metric] + other_metrics

    for other_metric in other_metrics:
        compare_metrics(baseline_metric, other_metric)

    representation = BLOSUMIndices(flatten=True)
    metric = BLOSUM62Score()
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
            result["percentile"] = (np.array(actual_distances) < result["closest_neighbor_distance"]).sum() / len(actual_distances)
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
            try:
                summary[k][key] = results_df[key].dropna(inplace=False).values.mean()
            except:
                import pdb; pdb.set_trace()

    df = pd.DataFrame.from_dict(summary, orient="index")
    print(df)
    df.to_csv("summary.csv")

    fig = go.Figure()
    for k in ks:
        fig.add_trace(go.Histogram(x=bucket_counts[k], name=k))
    fig.update_layout(title=dict(text="Bucket size distribution"), xaxis_title=dict(text="# sequences"), yaxis_title=dict(text="# buckets"), barmode="stack")
    fig.show()

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df.index, y=df["closest_neighbor_distance"], mode="lines", name="closest_neighbor_distance"))
    # fig.add_trace(go.Scatter(x=df.index, y=df["actual_neighbor_distance"], mode="lines", name="actual_neighbor_distance"))
    # fig.add_trace(go.Scatter(x=df.index, y=df["avg_distance"], mode="lines", name="avg_distance"))
    # fig.show()

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df.index, y=df["query_time"], mode="lines", name="query_time"))
    # fig.add_trace(go.Scatter(x=df.index, y=df["compute_time"], mode="lines", name="compute_time"))
    # fig.show()

    # px.line(df, y=["closest_neighbor_distance", "actual_neighbor_distance"], x="index")