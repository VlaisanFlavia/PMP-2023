import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

dummy_data_500 = np.loadtxt('dummy.csv')
x_2 = dummy_data_500[:, 0]
y_2 = dummy_data_500[:, 1]

# Change order to 5
order = 5
x_2p = np.vstack([x_2**i for i in range(1, order+1)])
x_2s = (x_2p - x_2p.mean(axis=1, keepdims=True)) / x_2p.std(axis=1, keepdims=True)
y_2s = (y_2 - y_2.mean()) / y_2.std()

# Plot the data
plt.scatter(x_2s[0], y_2s)
plt.xlabel('x')
plt.ylabel('y')

# Linear Model with 500 data points
with pm.Model() as model_l_500:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10)
    ε = pm.HalfNormal('ε', 5)
    μ = α + β * x_2s[0]
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_2s)
    idata_l_500 = pm.sample(2000, return_inferencedata=True)

# Polynomial Model with 500 data points
with pm.Model() as model_p_500:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_2s)
    idata_p_500 = pm.sample(2000, return_inferencedata=True)

# Polynomial Model with sd=100 and 500 data points
with pm.Model() as model_p_sd_100_500:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=100, shape=order)  # Change sd to 100
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_2s)
    idata_p_sd_100_500 = pm.sample(2000, return_inferencedata=True)

# Polynomial Model with different sd for each beta and 500 data points
with pm.Model() as model_p_sd_array_500:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)  # Change sd to an array
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_2s)
    idata_p_sd_array_500 = pm.sample(2000, return_inferencedata=True)


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
