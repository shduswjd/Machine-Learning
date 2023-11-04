import numpy as np

def regularize_cov(covariance, epsilon):
  D = covariance.shape[0]
  I = np.identity(D)* epsilon
  regularized_cov = covariance + I[:, :, np.newaxis]

  return regularized_cov