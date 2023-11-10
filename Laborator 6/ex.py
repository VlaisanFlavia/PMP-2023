import pymc as pm
import numpy as np
import arviz as az

Y_values = [0, 5, 10]
θ_values = [0.2, 0.5]


prior_n = 10

with pm.Model() as model:
    n = pm.Poisson('n', mu=prior_n)

    for i in range(len(Y_values)):
        for j in range(len(θ_values)):
            likelihood.observed = Y_values[i]
            trace = pm.sample(1000, tune=1000)

            az.plot_posterior(trace)
