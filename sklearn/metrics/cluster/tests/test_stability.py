import numpy as np

from sklearn.utils.testing import (
    assert_almost_equal, assert_equal, assert_array_equal)

from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics.cluster.stability import (
    adjacency_matrix, _one_stability_measure, stability)
from sklearn.datasets import make_blobs


def test_adjacency_matrix():
    assi = [0, 0, 1, 1]
    adj_matrix = np.array([[1, 1, 0, 0], [1, 1, 0, 0],
                           [0, 0, 1, 1], [0, 0, 1, 1]])
    found_adj = adjacency_matrix(assi)
    assert_array_equal(found_adj, adj_matrix)


def test_one_stability_measure():
    X = np.arange(10) < 5
    X.reshape((10, 1))

    # test perfect clustering has 1 stability
    class SameCluster(object):
        def set_params(self, *args, **kwargs):
            pass

        def fit_predict(self, X):
            return X
    same_clusters = SameCluster()
    assert_almost_equal(_one_stability_measure(same_clusters, X, .8), 1)


def test_stability():
    X, _ = make_blobs(90, centers=np.array([[-2, -2], [2, 0], [-2, 2]]),
                      random_state=0)
    cluster_estimator = KMeans()
    assert_equal(stability(X, cluster_estimator, k_max=6,
                           nb_draw=10, random_state=0), 3)
