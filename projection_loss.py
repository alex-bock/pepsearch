
import plotly.express as px
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr

from pephash.library import Library
from pephash.representations import BLOSUMIndices
from pephash.projection import GaussianRandomProjection


if __name__ == "__main__":

    m = 50
    library = Library(n=100, m=100)
    model = BLOSUMIndices(flatten=True)

    X = library.represent(model)
    X_dists = distance_matrix(X, X)
    px.imshow(X_dists).show()

    for k in (8, 10, 12, 14):

        projection = GaussianRandomProjection(d=m * model.dim, k=k)
        projection.fit(X)
        X_p = projection(X)

        X_p_dists = distance_matrix(X_p, X_p)
        print(k, spearmanr(X_dists.flatten(), X_p_dists.flatten()).statistic)
        px.imshow(X_p_dists).show()
