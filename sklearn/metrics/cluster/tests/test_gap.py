import numpy as np

from sklearn.utils.testing import (assert_true, assert_equal)

from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics.cluster.gap_statistic import (normal_distortion,
                                                   gap_statistic)
from sklearn.datasets import make_blobs


def test_normal_distortion():
    class BogusCluster(object):
        def fit_predict(self, points):
            n = len(points)
            mid = n / 2
            return [int(i < mid) for i in range(n)]
    var_1_data = np.asarray([[0, 0]] * 100 + [[2, 2]] * 100)
    mean_dist = np.mean(normal_distortion(
        var_1_data, BogusCluster(), nb_draw=10, random_state=0))
    # Expected mean dist is 1.
    # After 100 tries, it should be between .90 and 1.1
    assert_true(mean_dist > .9)
    assert_true(mean_dist < 1.1)


def test_gap_statistic():
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X, _ = make_blobs(90, centers=np.array([[-2, -2], [2, 0], [-2, 2]]),
                      random_state=0)
    cluster_estimator = KMeans()
    assert_equal(gap_statistic(X, cluster_estimator, k_max=6, nb_draw=10,
                               random_state=0, draw_model='normal'), 3)
    assert_equal(gap_statistic(X, cluster_estimator, k_max=6, nb_draw=10,
                               random_state=0, distortion_meth='cityblock'), 3)
