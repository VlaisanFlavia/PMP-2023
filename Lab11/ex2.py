import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

means = [5, 0, -5]
std_devs = [2, 2, 2]
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
mix = np.concatenate([
    np.random.normal(means[i], std_devs[i], n_cluster[i]) for i in range(len(means))
])

mix = mix.reshape(-1, 1)

models = [GaussianMixture(n_components=n, random_state=0).fit(mix) for n in range(2, 5)]

plt.figure(figsize=(12, 8))
plt.hist(mix, bins=30, density=True, alpha=0.7, color='g')

for i, model in enumerate(models, 2):
    x = np.linspace(-15, 15, 1000)
    pdf = np.exp(model.score_samples(x.reshape(-1, 1)))
    plt.plot(x, pdf, label=f'Mixtura cu {i} componente')

plt.title('Mixtura de distribuții Gaussiene cu 2, 3 și 4 componente')
plt.xlabel('Valoare')
plt.ylabel('Densitate')
plt.legend()
plt.show()