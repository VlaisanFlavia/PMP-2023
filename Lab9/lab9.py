#NU AM REZOLVAT PROBLEMA CU EROAREA LA PYMC...

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
fisier=files.upload()

# Load data
admission_data = pd.read_csv('Admission.csv')
gre_scores = admission_data['GRE']
gpa_scores = admission_data['GPA']
admission_outcome = admission_data['Admission']

logistic_model = pymc.Model()

with pm.Model() as logistic_model:
    beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=10)

    pi = pm.Deterministic('pi', pm.math.sigmoid(beta_0 + beta_1 * gre_scores + beta_2 * gpa_scores))

    admission_likelihood = pm.Bernoulli('admission_likelihood', p=pi, observed=admission_outcome)

with logistic_model:
    trace = pm.sample(2000, tune=1000)

beta_0_samples = trace['beta_0']
beta_1_samples = trace['beta_1']
beta_2_samples = trace['beta_2']
decision_boundary = -beta_0_samples / beta_1_samples
az.plot_hdi(np.mean(decision_boundary, axis=0), hdi_prob=0.94, color='gray', fill_kwargs={'alpha': 0.3})
plt.scatter(gre_scores, gpa_scores, c=admission_outcome, cmap='viridis')
plt.xlabel('GRE Score')
plt.ylabel('GPA')
plt.title('Decision Boundary and 94% HDI')
plt.show()

gre_new = 550
gpa_new = 3.5
pi_new = pm.Deterministic('pi_new', pm.math.sigmoid(beta_0_samples + beta_1_samples * gre_new + beta_2_samples * gpa_new))
hdi_90_new = hpd(pi_new, credible_interval=0.90)

print(f'Intervalul HDI pentru probabilitatea de admitere: {hdi_90_new}')

gre_new = 500
gpa_new = 3.2
pi_new = pm.Deterministic('pi_new', pm.math.sigmoid(beta_0_samples + beta_1_samples * gre_new + beta_2_samples * gpa_new))
hdi_90_new = hpd(pi_new, credible_interval=0.90)

print(f'Intervalul HDI pentru probabilitatea de admitere: {hdi_90_new}')