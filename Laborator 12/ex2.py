import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside_circle.sum() / N
    error = np.abs(pi_estimate - np.pi)
    return error

# Valorile N
Ns = [100, 1000, 10000]

# Rulez codul de mai multe ori pentru fiecare N și salvez erorile
errors = []
for N in Ns:
    error_per_N = [estimate_pi(N) for _ in range(100)]  
    errors.append(error_per_N)

# Media și deviația standard a erorilor
mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

plt.errorbar(Ns, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
plt.xscale('log')
plt.xlabel('Numărul de puncte (N)')
plt.ylabel('Eroare în estimarea lui π')
plt.title('Estimarea lui π cu diferite N și erori asociate')
plt.show()
