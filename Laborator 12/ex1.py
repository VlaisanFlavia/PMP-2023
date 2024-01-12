import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def posterior_grid(grid_points=50, heads=6, tails=9, prior_type='uniform'):
    grid = np.linspace(0, 1, grid_points)

    if prior_type == 'uniform':
        prior = np.repeat(1/grid_points, grid_points) 
    elif prior_type == 'conditioned':
        prior = (grid <= 0.5).astype(int)
    elif prior_type == 'absolute_difference':
        prior = np.abs(grid - 0.5)
    elif prior_type == 'custom_distribution':
        prior = stats.norm.pdf(grid, loc=0.5, scale=0.2)
    else:
        raise ValueError("Invalid prior_type. Choose 'uniform', 'conditioned', 'absolute_difference', or 'custom_distribution'.")

    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    
    return grid, posterior

data = np.repeat([0, 1], (20, 10)) 
points = 100 

h = data.sum()
t = len(data) - h

prior_type = 'uniform'

grid, posterior = posterior_grid(points, h, t, prior_type)

plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}, Prior = {prior_type}')
plt.yticks([])
plt.xlabel('θ')
plt.show()
