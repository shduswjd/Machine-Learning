import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def regularize_cov(covariance, epsilon):
    D = covariance.shape[0]
    I = np.identity(D) * epsilon
    regularized_cov = covariance + I[:, :, np.newaxis]
    return regularized_cov

def EStep(means, covariances, weights, X, epsilon):
    N, D = X.shape
    K = len(means)
    gamma = np.zeros((N, K))
    recov = regularize_cov(covariances, epsilon)

    for n in range(N):
        x = X[n]

        sum_weights = 0
        for k in range(K):
            mean = means[k]
            cov = recov[:, :, k]
            weight = weights[k]
            x_m = x - mean
            exp = np.exp((-0.5) * np.sum(x_m * np.dot(np.linalg.inv(cov), x_m.T)))
            det = np.linalg.det(cov)
            total = weight * (exp / np.sqrt((2 * np.pi) ** D * det))
            sum_weights += total

        for k in range(K):
            mean = means[k]
            cov = recov[:, :, k]
            weight = weights[k]
            x_m = x - mean
            exp = np.exp((-0.5) * np.sum(x_m * np.dot(np.linalg.inv(cov), x_m.T)))
            det = np.linalg.det(cov)
            total = weight * (exp / np.sqrt((2 * np.pi) ** D * det))
            gamma[n, k] = total / sum_weights

    return gamma

def MStep(gamma, X):
    N, D = X.shape
    K = gamma.shape[1]
    means = np.zeros((K, D))
    covariances = np.zeros((D, D, K))
    weights = np.zeros(K)
    logLikelihood = 0

    for k in range(K):
        n_j = np.sum(gamma[:, k])
        weights[k] = n_j / N
        means[k, :] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / n_j
        x_m = X - means[k]
        covariances[:, :, k] = np.dot((x_m * gamma[:, k][:, np.newaxis]).T, x_m) / n_j
        det = np.linalg.det(covariances[:, :, k])
        total = weights[k] / (np.sqrt((2 * np.pi) ** D * det))
        likelihood = total

    logLikelihood += np.sum(np.log(likelihood))

    return weights, means, covariances, logLikelihood

def initialize_params(data, K, epsilon):
    weights = np.ones(K) / K
    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_
    covariances = np.zeros((data.shape[1], data.shape[1], K))

    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(data.shape[1]) * min_dist

    return weights, means, covariances

def estGaussMixEM(data, K, n_iters, epsilon):
    N, D = data.shape
    weights, means, covariances = initialize_params(data, K, epsilon)
    recovariances = regularize_cov(covariances, epsilon)

    for i in range(n_iters):
        gamma = EStep(means, recovariances, weights, data, epsilon)
        weights, means, recovariances, logLikelihood = MStep(gamma, data)

    return weights, means, recovariances

# Example usage
# data1 = np.loadtxt('data1')
# K = 3
# n_iters = 5
# epsilon = 0.0001
# weights, means, covariances = estGaussMixEM(data1, K, n_iters, epsilon)
# print(weights)
# print(means)
# print(covariances)
# print(logLikelihood)


