import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

intervale = [(9, 12), (12, 15), (15, 18)]
nr_persoane = 5

with pm.Model() as model:
    lambda_param = pm.Exponential("lambda", lam=1)
    
    for i, interval in enumerate(intervale):
        interval_obs = nr_persoane[(i * 60):(interval[1] * 60)]
        trafic = pm.Poisson(f"trafic_interval_{i}", mu=lambda_param, observed=interval_obs)
    
    trace = pm.sample(1000, tune=1000, cores=2)  


az.summary(trace)
az.plot_trace(trace)