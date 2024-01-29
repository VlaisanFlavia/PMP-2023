import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

#1.a
# Încărcăm setul de date Titanic.csv
data = pd.read_csv('Titanic.csv')

# Gestionăm valorile lipsă
data.fillna(method='ffill', inplace=True) 

# Transformarea variabilelor
# Am ales să transformăm cuvântul de bază male/female -> 0/1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Definim variabilele dependente și independente
# Vom utiliza doar variabilele 'Pclass', 'Age' și 'Survived' pentru acest model
X = np.array(titanic_data[['Pclass', 'Age']])
y = np.array(titanic_data['Survived'])

# Definim modelul PyMC
with pm.Model() as model:
    # Coeficienții pentru regresia logistică
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    
    # Logitul (suma ponderată a variabilelor independente)
    logit = beta[0] * X[:, 0] + beta[1] * X[:, 1]
    
    # Probabilitatea de a supraviețui
    p = pm.invlogit(logit)
    
    # Variabila observată (Survived)
    observed = pm.Bernoulli('observed', p=p, observed=y)
    
    # Efectuăm inferențe
    trace = pm.sample(1000, tune=1000)

# Vizualizăm rezultatele
az.plot_trace(trace)
plt.show()

#1.c:
#Vom analiza mai intai coeficinetii Pclass si Age pentru a trage o concluzie 
# Vizualizăm rezultatele
pm.summary(trace)
#Variabila 'Pclass' are o influență mai mare asupra rezultatului (șansele de supraviețuire) decât variabila 'Age', conform rezultatului

#1.d:
# Definesc varsta și clasa pasagerului pentru care trebuie să calculez probabilitatea de supraviețuire
varsta = 30
clasa = 2

# Filtrăm pentru a selecta doar iterațiile asociate vârstei și clasei specificate
prob_supravietuire = trace['observed'][:, (X[:, 0] == clasa) & (X[:, 1] == varsta)]


# Calculăm intervalul HDI
hdi_90 = az.hdi(prob_supravietuire, hdi_prob=0.9)

print("Intervalul HDI pentru probabilitatea de supraviețuire a unui pasager de 30 de ani din clasa a 2-a:")
print(hdi_90)
