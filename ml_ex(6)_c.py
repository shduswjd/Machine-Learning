import numpy as np

def MStep(gamma, X):
  N, D = X.shape
  K = gamma.shape[1]
  means = np.zeros((K, D))
  covariances = np.zeros((D, D, K))
  weights = np.zeros(K)

  logLikelihood = 0

  for k in range(K):
    # gamma sum
    n_j = np.sum(gamma[:, k])

    # weights
    weights[k] = n_j/ N

    # means
    means[k] = np.sum(gamma[:,k][:, np.newaxis] * X, axis=0) / n_j

    # covariances
    x_m = X - means[k]
    covariances[:,:,k] = np.dot((x_m * gamma[:, k][:, np.newaxis]).T, x_m)/n_j


    exp = np.exp(((-1/2) * np.sum(np.dot(np.dot(x_m, np.linalg.inv(covariances[:,:,k])), x_m.T))))
    det = np.linalg.det(covariances[:,:,k])
    total = weights[k] * (exp/np.sqrt((2*np.pi)**D * det)) # N(k)
    likelihood = total
  logLikelihood += np.sum(np.log(likelihood))

  return [weights, means, covariances, logLikelihood]

