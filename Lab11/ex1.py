import numpy as np
import matplotlib.pyplot as plt

means = [5, 0, -5] 
std_devs = [2, 2, 2] 
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)

mix = np.concatenate([
    np.random.normal(means[i], std_devs[i], n_cluster[i]) for i in range(len(means))
])

plt.hist(mix, bins=30, density=True, alpha=0.7, color='g')
plt.title('Mitură de trei distribuții Gaussiene')
plt.xlabel('Valoare')
plt.ylabel('Densitate')
plt.show()