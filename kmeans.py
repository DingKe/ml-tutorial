from __future__ import print_function

import numpy as np

import pytest


EPS = 1e-8


def kmeans_cluster(x, k, max_iter=10, threshold=1e-3, verbose=False):
    # init
    centers = np.zeros([k, x.shape[-1]])
    for i in range(k):
        total_num = len(x)
        chosen_num = max(1, total_num / k)
        random_ids = np.random.choice(total_num, chosen_num, replace=False)
        centers[i, :] = np.mean(x[random_ids])

    cur_total_dist = np.float('inf')
    dist = np.zeros([k, len(x)])
    for i in range(max_iter):
        for j in range(k):
            for m, p in enumerate(x):
                dist[j, m] = np.mean((p - centers[j]) ** 2)

        min_idx = np.argmin(dist, 0)
        for j in range(k):
            ele = [x[m] for m, idx in enumerate(min_idx) if idx == j]
            centers[j, :] = np.mean(ele)

        total_dist = 0
        for j in range(k):
            dist_j = [np.mean((x[m] - centers[j]) ** 2)
                      for m, idx in enumerate(min_idx) if idx == j]
            total_dist += sum(dist_j)

        cur_total_dist, last_total_dist = total_dist, cur_total_dist
        if verbose:
            print('Iter: {}, total dist: {}'.format(i, total_dist))

        if last_total_dist - cur_total_dist < threshold:
            break

    for j in range(k):
        for m, p in enumerate(x):
            dist[j, m] = np.mean((p - centers[j]) ** 2)
    min_idx = np.argmin(dist, 0)

    return centers, min_idx


def test_kmeans_cluster():
    np.random.seed(1111)
    dims = range(1, 5)
    ks = range(2, 5)

    for dim in dims:
        for k in ks:
            print('Dim {}, K {}'.format(dim, k))
            x = np.random.random([100, dim])
            kmeans_cluster(x, k, max_iter=10, threshold=1e-2, verbose=True)


def demo():
    import matplotlib.pyplot as plt

    np.random.seed(1111)
    dim, k = 2, 2

    # generate data
    num = 50
    mean1 = np.zeros(dim)
    mean2 = np.ones(dim) * 2
    cov1 = np.eye(dim)
    cov2 = np.eye(dim) * 0.5

    x1 = np.random.multivariate_normal(mean1, cov1, [num, ])
    x2 = np.random.multivariate_normal(mean2, cov2, [num, ])
    x = np.concatenate([x1, x2], 0)

    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='g')

    centers, _ = kmeans_cluster(x, k, max_iter=10, threshold=1e-4)
    plt.scatter(centers[:, 0], centers[:, 1], c='y', s=500, marker='^')


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
