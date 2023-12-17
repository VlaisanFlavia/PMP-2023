import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

dummy_data_500 = np.loadtxt('dummy.csv')
x_2 = dummy_data_500[:, 0]
y_2 = dummy_data_500[:, 1]

# Cubic Model with 500 data points
order_cubic = 3
x_3p = np.vstack([x_2**i for i in range(1, order_cubic+1)])
x_3s = (x_3p - x_3p.mean(axis=1, keepdims=True)) / x_3p.std(axis=1, keepdims=True)

with pm.Model() as model_l_500:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10)
    ε = pm.HalfNormal('ε', 5)
    μ = α + β * x_2s[0]
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_2s)
    idata_l_500 = pm.sample(2000, return_inferencedata=True, log_likelihood=True)

# Polynomial Model with 500 data points
with pm.Model() as model_p_500:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_2s)
    idata_p_500 = pm.sample(2000, return_inferencedata=True, log_likelihood=True)

# WAIC and LOO for each model
waic_l_500 = pm.waic(idata_l_500)
waic_p_500 = pm.waic(idata_p_500)
waic_p_sd_100_500 = pm.waic(idata_p_sd_100_500)
waic_p_sd_array_500 = pm.waic(idata_p_sd_array_500)
waic_cubic_500 = pm.waic(idata_cubic_500)

loo_l_500 = pm.loo(idata_l_500)
loo_p_500 = pm.loo(idata_p_500)
loo_p_sd_100_500 = pm.loo(idata_p_sd_100_500)
loo_p_sd_array_500 = pm.loo(idata_p_sd_array_500)
loo_cubic_500 = pm.loo(idata_cubic_500)

# Print WAIC and LOO for comparison
print("WAIC - Linear Model (500 data points):", waic_l_500.waic)
print("LOO - Linear Model (500 data points):", loo_l_500.loo)
print("WAIC - Quadratic Model (500 data points):", waic_p_500.waic)
print("LOO - Quadratic Model (500 data points):", loo_p_500.loo)
print("WAIC - Quadratic Model (sd=100, 500 data points):", waic_p_sd_100_500.waic)
print("LOO - Quadratic Model (sd=100, 500 data points):", loo_p_sd_100_500.loo)
print("WAIC - Quadratic Model (sd array, 500 data points):", waic_p_sd_array_500.waic)
print("LOO - Quadratic Model (sd array, 500 data points):", loo_p_sd_array_500.loo)
print("WAIC - Cubic Model (500 data points):", waic_cubic_500.waic)
print("LOO - Cubic Model (500 data points):", loo_cubic_500.loo)



x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
y_l_post = α_l_post + β_l_post * x_new
plt.plot(x_new, y_l_post, 'C1', label='linear model')


α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.show()