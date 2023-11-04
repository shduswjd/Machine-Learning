import numpy as np

def EStep(means, weights, covariances, X):
    N, D = X.shape
    K = len(means)
    log_likelihood = 0

    gamma = np.zeros((N, K))

    for n in range(N):
        sample = X[n]
        likelihood = 0

        sum = 0
        for k in range(K):
          mean = means[k]
          covariance = covariances[:,:,k]
          weight = weights[k]
          x_m = sample - mean
          exp = np.exp(((-1/2) * np.dot(np.dot(x_m, np.linalg.inv(covariance)), x_m.T)))
          det = np.linalg.det(covariance)
          total = weight * (exp/np.sqrt((2*np.pi)**D * det)) # N(k)
          sum += total


        for k in range(K):
            mean = means[k]
            covariance = covariances[:,:,k]
            weight = weights[k]

            x_m = sample - mean
            exp = np.exp(((-1/2) * np.dot(np.dot(x_m, np.linalg.inv(covariance)), x_m.T)))
            det = np.linalg.det(covariance)
            total = weight * (exp/np.sqrt((2*np.pi)**D * det)) # N(k)
            likelihood += total
            gamma[n, k] = total/sum

        log_likelihood += np.log(likelihood)

    return [log_likelihood, gamma]