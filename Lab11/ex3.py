import arviz as az

# Datele
data = {'x': mix}

model_names = ['Model 2', 'Model 3', 'Model 4']
az_data = []

for i, model in enumerate(models, 2):
    with model:
        trace = model.sample(1000, random_seed=0)  # Sample from the posterior

    az_trace = az.from_pymc3(trace, coords={'model': [f'Model {i}']})
    az_data.append(az_trace)

waic_results = az.compare(az_data, ic='waic')
loo_results = az.compare(az_data, ic='loo')

print("WAIC Comparison:")
print(waic_results)

print("\nLOO Comparison:")
print(loo_results)

az.plot_compare(waic_results)
az.plot_compare(loo_results)

plt.show()
