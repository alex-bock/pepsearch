
import json
import os
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pephash.library import Library
from pephash.metrics import compare_metrics, BLOSUM62Score, BLOSUMIndexDistance
from pephash.experiment import MurmurHashExperiment, get_ground_truth


if __name__ == "__main__":

    np.random.seed(1047)

    n = 1000000
    m = 50

    # library = Library(n=n, m=m)
    query_set = Library(n=10, m=m)
    print(query_set.seqs)
    exit()

    ks = [8, 9, 10, 11, 12, 13, 14]

    metric = BLOSUM62Score()

    if os.path.exists(f"results/m={m}/gt.json"):
        with open(f"results/m={m}/gt.json", "r") as f:
            ground_truth = json.load(f)
        ground_truth_df = pd.read_csv(f"results/m={m}/gt.csv")
    else:
        exit()

    summary = dict()
    bucket_counts = dict()

    for k in ks:

        pprint(
            {
                "metric": metric,
                "k": k
            }
        )

        experiment = MurmurHashExperiment(k=k)

        experiment.load_library(library, library.seqs.values)
        bucket_counts[k] = experiment.quantify_table()

        results = experiment.evaluate(query_set, query_set.seqs.values, metric)
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