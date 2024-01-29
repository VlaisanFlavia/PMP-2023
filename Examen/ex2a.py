import numpy as np
from scipy.stats import geom

#2.a:
# Definirea parametrilor pentru variabilele aleatoare X și Y
theta_X = 0.3
theta_Y = 0.5

# Numărul de iterații Monte Carlo
num_iteratii = 10000

# Generăm 10000 de perechi de variabile aleatoare X și Y
X = geom.rvs(theta_X, size=num_iteratii)
Y = geom.rvs(theta_Y, size=num_iteratii)

# Calculăm probabilitatea că X > Y^2 pentru fiecare pereche și estimăm probabilitatea medie
probabilitati = (X > Y ** 2).mean()

# Afișăm rezultatul
print("Aproximarea probabilității P(X > Y^2):", probabilitati)

#Rezultate: Aproximarea probabilității P(X > Y^2): 0.4144