# -*- coding: utf-8 -*-
# 使用EM算法求解高斯混合模型
import numpy as np
def gauss(y, mu, sigma, alpha):
    N = y.shape[1]
    K = mu.shape[1]
    alpha = np.tile(alpha, (N, 1))
    mu = np.tile(mu, (N, 1))
    sigma = np.tile(sigma, (N, 1))
    Y = np.tile(y.reshape(N, 1), (1, K))
    gamma = alpha * 1 / (np.sqrt(sigma * 2 * np.pi)) * np.exp(-(Y - mu) ** 2 / (2 * sigma))
    linesum = np.sum(gamma, axis=1, keepdims=True)
    gamma /= linesum
    assert gamma.shape == (N, K)
    return gamma

def mixed_gauss(y, K, epsilon = 1e-4):
    alpha = np.random.rand(1, K)
    mu = np.random.rand(1, K)
    sigma = np.random.rand(1, K)
    #对系数进行归一化
    alpha = alpha / np.sum(alpha)
    y = y.reshape((1, -1))
    N = y.shape[1]
    while True:
        gamma = gauss(y, mu, sigma, alpha)
        new_alpha = np.sum(gamma, axis=0, keepdims=True) / N
        new_mu = y.dot(gamma) / np.sum(gamma, axis=0, keepdims=True)
        new_sigma = np.sum(gamma * ((y.T - mu) ** 2), axis=0, keepdims=True) \
                    / np.sum(gamma, axis=0, keepdims=True)
        assert new_alpha.shape == (1, K)
        assert new_mu.shape == (1, K)
        assert new_sigma.shape == (1, K)
        if np.abs(alpha - new_alpha).max() < epsilon and \
                np.abs(sigma - new_sigma).max() < epsilon and \
                np.abs(mu - new_mu).max() < epsilon:
            break
        alpha = new_alpha
        mu = new_mu
        sigma = new_sigma
    return alpha, mu, sigma
if __name__ == '__main__':
    mu1, mu2, mu3 = -0.4, 0.5, 2.2
    sigma1, sigma2, sigma3 = 1.0, 0.7, 1.5
    alpha1, alpah2, alpha3 = 0.1, 0.6, 0.3
    N = 1000
    y = alpha1 * np.random.normal(mu1, sigma1, N) \
        + alpah2 * np.random.normal(mu2, sigma2, N) \
        + alpha3 * np.random.normal(mu3, sigma3, N)
    alpha, mu, sigma = mixed_gauss(y, 3)
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3).fit(y.reshape(-1,1))