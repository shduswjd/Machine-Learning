import numpy as np
import matplotlib.pyplot as plt

# p(x) = k/NV = k/ (N * c * R)
# c : volume of the unit sphere in D dimensions 
# c = pi**(D/2) / (D/2)!
# R : distance b/t the estimation point and its k-th closest neighbor
def knn(sample, K):
  pos = np.arange(-5, 5.0, 0.1)
  N = len(pos)
  density = np.zeros(len(pos))

  for i, val in enumerate(pos):
    distance = np.abs(sample - val)
    sort_dis = np.sort(distance)
    k_near = sort_dis[:K]

    density[i] = K/(N * 2 * np.mean(k_near))

  est_density = np.stack((pos, density), axis = 1)

  return est_density

def gauss1D(m, v, N, w):
  pos = np.arange(-w, w - w / N, 2 * w / N)
  insE = -0.5 * ((pos - m) / v) ** 2
  norm = 1 / (v * np.sqrt(2 * np.pi))
  res = norm * np.exp(insE)
  realDensity = np.stack((pos, res), axis=1)
  return realDensity


samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
realDensity = gauss1D(0, 1, 100, 5)

k = 30
estDensity = knn(samples, k)

# Plot the distributions
plt.subplot(2, 1, 2)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KNN Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()
plt.show()
