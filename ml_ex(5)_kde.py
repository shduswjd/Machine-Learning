import numpy as np
import matplotlib.pyplot as plt

def kde(samples, h):
  pos = np.arange(-5, 5, 0.1)
  density_lst = np.zeros(len(pos))

  for i in range(len(pos)):
    x = pos[i]
    sum = 0

    for j in range(len(pos)):
      u = x - samples[j]
      exp = np.exp(-((np.abs(u))**2)/(2 * h**2))
      kernel_func = exp / np.sqrt(2 * np.pi * h**2)
      sum += kernel_func

    density_lst[i] = sum/len(pos)

  est_density = np.stack((pos, density_lst), axis = 1)

  return est_density

def gauss1D(m, v, N, w):
    pos = np.arange(-w, w - w / N, 2 * w / N)
    insE = -0.5 * ((pos - m) / v) ** 2
    norm = 1 / (v * np.sqrt(2 * np.pi))
    res = norm * np.exp(insE)
    realDensity = np.stack((pos, res), axis=1)
    return realDensity

h = 0.3
realDensity = gauss1D(0, 1, 100, 5)
samples = np.random.normal(0, 1, 100)

# Estimate the probability density using the KDE
estDensity = kde(samples, h)

# plot results
plt.subplot(2, 1, 1)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KDE Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()