import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate data from two Gaussian distributions
np.random.seed(42)
data1 = np.random.normal(loc=5, scale=1, size=150)  # N(5, 1)
data2 = np.random.normal(loc=10, scale=np.sqrt(2), size=150)  # N(10, 2)
data = np.concatenate([data1, data2])

# Number of clusters and initialize parameters
K = 2
n = len(data)

# Initial parameters
mu = np.array([6, 9])  # Initial means
sigma = np.array([1.0, 1.5])  # Initial variances
pi = np.array([0.5, 0.5])  # Initial mixing coefficients

# Store responsibilities
r = np.zeros((n, K))

# EM Algorithm
max_iter = 100
tolerance = 1e-6

for iteration in range(max_iter):
    # E-step: Compute responsibilities
    for k in range(K):
        r[:, k] = pi[k] * norm.pdf(data, mu[k], np.sqrt(sigma[k]))
    r = r / r.sum(axis=1, keepdims=True)  # Normalize responsibilities

    # M-step: Update parameters
    N_k = r.sum(axis=0)
    mu_new = np.sum(r * data[:, np.newaxis], axis=0) / N_k
    sigma_new = np.sum(r * (data[:, np.newaxis] - mu_new) ** 2, axis=0) / N_k
    pi_new = N_k / n

    # Check convergence
    if np.allclose(mu, mu_new, atol=tolerance) and np.allclose(sigma, sigma_new, atol=tolerance):
        break

    mu, sigma, pi = mu_new, sigma_new, pi_new

# Final parameters
mu, sigma, pi

# Plot the results
x = np.linspace(min(data) - 1, max(data) + 1, 1000)
pdf1 = pi[0] * norm.pdf(x, mu[0], np.sqrt(sigma[0]))
pdf2 = pi[1] * norm.pdf(x, mu[1], np.sqrt(sigma[1]))
plt.hist(data, bins=30, density=True, alpha=0.6, color='gray', label='Data histogram')
plt.plot(x, pdf1, label=f'N({mu[0]:.2f}, {sigma[0]:.2f})', lw=2)
plt.plot(x, pdf2, label=f'N({mu[1]:.2f}, {sigma[1]:.2f})', lw=2)
plt.plot(x, pdf1 + pdf2, label='Combined', lw=2, linestyle='--')
plt.legend()
plt.title('EM Algorithm for GMM')
plt.xlabel('Data')
plt.ylabel('Density')
plt.show()
