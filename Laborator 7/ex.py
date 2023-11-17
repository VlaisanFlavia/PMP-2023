import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('auto-mpg.csv')

sns.scatterplot(x='CP', y='mpg', data=df)
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.title('Relația dintre CP și mpg')
plt.show()

with pm.Model() as model:
    cp = pm.Data('cp', df['CP'])
    mpg = pm.Data('mpg', df['mpg'])

    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + beta * cp
    sigma = pm.Exponential('sigma', 1)

    likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=mpg)


with model:
    trace = pm.sample(2000, tune=1000)

pm.summary(trace).round(2)

sns.scatterplot(x='CP', y='mpg', data=df)
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.title('Regresia dintre CP și mpg cu HDI')

pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(df['CP'].min(), df['CP'].max(), 100), color='lightblue', alpha=0.5)

plt.show()
