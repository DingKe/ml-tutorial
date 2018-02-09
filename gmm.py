from __future__ import print_function

import numpy as np
from scipy.stats import multivariate_normal

import pytest
from numpy.testing import assert_allclose


EPS = 1e-8


class Gauss(object):
    '''
    '''

    def __init__(self, dim, mean=None, cov=None):
        self.dim = dim

        if mean is None:
            self.mean = np.zeros(dim)
        else:
            assert len(mean) == dim, "Dim not match"
            self.mean = mean

        if cov is None:
            self.cov = np.eye(dim)
        else:
            self.cov = cov

        self.rv = multivariate_normal(self.mean, self.cov)

    def update(self, mean, cov):
        self.mean, self.cov = mean, cov
        self.rv = multivariate_normal(self.mean, self.cov)

    def pdf(self, x):
        return self.rv.pdf(x)

    def __call__(self, x):
        return self.pdf(x)


class GMM(object):
    '''
    '''

    def __init__(self, gauss, weight=[]):
        self.gauss = gauss
        self.weight = weight or np.ones(len(gauss)) / len(gauss)

    @property
    def k(self):
        return len(self.gauss)

    def pdf(self, x):
        return sum([self.weight[i] * g(x) for i, g in enumerate(self.gauss)])

    def __call__(self, x, i=None):
        if i is None:
            return self.pdf(x)
        else:
            return self.weight[i] * self.gauss[i](x)

    def __getitem__(self, i):
        assert i < self.k, 'Out of Index'
        return self.gauss[i]

    def llk(self, x):
        return np.mean([np.log(self.pdf(e)) for e in x])


def em_step(gmm, x):
    num = len(x)
    dim = x.shape[-1]
    k = gmm.k

    gamma = np.zeros((k, num))

    # E
    for i in range(k):
        for j in range(num):
            gamma[i][j] = gmm(x[j], i)
    gamma /= np.sum(gamma, 0)

    # M
    gmm.weight = np.sum(gamma, 1) / num
    for i in range(k):
        mean = np.average(x, axis=0, weights=gamma[i])
        cov = np.zeros((dim, dim))
        for j in range(num):
            delta = x[j] - mean
            cov[:] += gamma[i][j] * np.outer(delta, delta)
        cov /= np.sum(gamma[i])
        cov += np.eye(dim) * EPS  # avoid singular
        gmm[i].update(mean, cov)

    return gmm


def prune_gmm(gmm, min_k=1):
    '''TODO: prune GMM components
    '''
    return gmm


def train_gmm(gmm, x, max_iter=100, threshold=1e-3, min_k=1):
    cur_llk = -np.float('inf')
    for i in range(max_iter):
        gmm = em_step(gmm, x)
        cur_llk, last_llk = gmm.llk(x), cur_llk

        print("Iter {}, log likelihood {}.".format(i, cur_llk))

        if cur_llk - last_llk < threshold:  # early stop
            break

        gmm = prune_gmm(gmm, min_k)


def test_gauss():
    dims = range(1, 5)

    # default
    for dim in dims:
        g = Gauss(dim)
        assert_allclose(g.mean, np.zeros(dim))
        assert_allclose(g.cov, np.eye(dim))

        x = np.random.random(dim)
        print(dim, g.pdf(x))

    # pass
    for dim in dims:
        mean = np.random.random(dim)
        cov = np.random.random([dim, dim])
        cov = np.matmul(cov, cov.T)

        g = Gauss(dim, mean, cov)
        assert_allclose(mean, g.mean)
        assert_allclose(cov, g.cov)

        x = np.random.random(dim)
        print(dim, g(x))


def test_gmm():
    dims = range(1, 5)
    ks = range(1, 5)

    for dim in dims:
        for k in ks:
            print('Dim {}, K {}'.format(dim, k))
            gs = []
            for i in range(k):
                mean = np.random.random(dim)
                cov = np.random.random([dim, dim])
                cov = np.matmul(cov, cov.T)
                gs.append(Gauss(dim, mean, cov))
            gmm = GMM(gs)

            assert k == gmm.k

            x = np.random.random(dim)
            assert gmm.pdf(x) == gmm(x)
            for i in range(k):
                print('Component {}, {}'.format(i, gmm(x, i)))
            print('log likelihood: %.2f' % gmm.llk(np.expand_dims(x, 0)))


def test_em_step():
    np.random.seed(1111)
    dims = range(1, 5)
    ks = range(1, 5)

    for dim in dims:
        for k in ks:
            print('Dim {}, K {}'.format(dim, k))
            gs = []
            for i in range(k):
                mean = np.random.random(dim)
                cov = np.random.random([dim, dim])
                cov = np.matmul(cov, cov.T)
                gs.append(Gauss(dim, mean, cov))
            gmm = GMM(gs)

            x = np.random.random([1000, dim])
            em_step(gmm, x)


def test_train_gmm():
    np.random.seed(1111)
    dims = range(1, 5)
    ks = range(1, 5)

    for dim in dims:
        for k in ks:
            print('Dim {}, K {}'.format(dim, k))
            gs = []
            for i in range(k):
                mean = np.random.random(dim)
                cov = np.random.random([dim, dim])
                cov = np.matmul(cov, cov.T)
                gs.append(Gauss(dim, mean, cov))
            gmm = GMM(gs)

            x = np.random.random([100, dim])
            train_gmm(gmm, x, threshold=1e-2)


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

    # init GMM
    gs = []
    ids = np.random.choice(range(len(x)), k, replace=False)
    for i in range(k):
        mean = x[ids[i]]
        cov = np.eye(dim)
        gs.append(Gauss(dim, mean, cov))
    gmm = GMM(gs)
    centers = np.stack([gmm[i].mean for i in range(gmm.k)])
    plt.scatter(centers[:, 0], centers[:, 1], c='b', s=50, marker='v')

    train_gmm(gmm, x, threshold=1e-4)
    centers = np.stack([gmm[i].mean for i in range(gmm.k)])
    plt.scatter(centers[:, 0], centers[:, 1], c='y', s=500, marker='^')


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
