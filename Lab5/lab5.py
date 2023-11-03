import pymc as pm
import numpy as np
import pandas as pd
import arviz as az


data = pd.read_csv("trafic.csv", sep='\t')
trafic_obs = data['nr. masini'].values


intervale = [(0, 7), (7, 8), (8, 16), (16, 19), (19, 24)]


with pm.Model() as model:
    lambda_param = pm.Exponential("lambda", lam=1)
    
    for i, interval in enumerate(intervale):
        interval_obs = trafic_obs[(i * 60):(interval[1] * 60)]
        trafic = pm.Poisson(f"trafic_interval_{i}", mu=lambda_param, observed=interval_obs)
    
    trace = pm.sample(1000, tune=1000, cores=2)  


az.summary(trace)
az.plot_trace(trace)


for i, interval in enumerate(intervale):
    lambda_interval = trace[f"lambda_interval_{i}"]
    print(f"Interval {i + 1} ({interval[0]} - {interval[1]}):")
    print(f"Valoare medie a lui lambda: {lambda_interval.mean()}")
    print(f"Interval de încredere al lui lambda: ({np.percentile(lambda_interval, 2.5)}, {np.percentile(lambda_interval, 97.5)})")


for i, interval in enumerate(intervale):
    lambda_interval = trace[f"lambda_interval_{i}"]
    print(f"Interval {i + 1} ({interval[0]} - {interval[1]}):")
    lower_bound = np.percentile(lambda_interval, 2.5)
    upper_bound = np.percentile(lambda_interval, 97.5)
    print(f"Capătul inferior: {lower_bound}")
    print(f"Capătul superior: {upper_bound}")
