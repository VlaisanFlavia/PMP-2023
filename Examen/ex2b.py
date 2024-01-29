#2.b:
# Definirea parametrilor pentru variabilele aleatoare X și Y
theta_X = 0.3
theta_Y = 0.5

# Numărul de iterații Monte Carlo si numarul de aproximari
num_iteratii = 10000
num_aproximari = 30

# Listă pentru a stoca rezultatele aproximărilor
rezultate = []

# Iterăm pentru a calcula P(X > Y^2) de 30 de ori
for _ in range(num_aproximari):
    # Generăm 10000 de perechi de variabile aleatoare X și Y
    X = geom.rvs(theta_X, size=num_iteratii)
    Y = geom.rvs(theta_Y, size=num_iteratii)
    
    # Calculăm probabilitatea că X > Y^2 pentru fiecare pereche și estimăm probabilitatea medie
    probabilitati = (X > Y ** 2).mean()
    
    # Adăugăm rezultatul la lista de rezultate
    rezultate.append(probabilitati)

# Calculăm media și deviația standard a rezultatelor
media_aproximari = np.mean(rezultate)
deviatia_standard = np.std(rezultate)

# Afișăm rezultatele
print("Media pentru aproximările P(X > Y^2):", media_aproximari)
print("Deviatia standard pentru aproximări:", deviatia_standard)

#Rezultate:
# Media pentru aproximările P(X > Y^2): 0.41647000000000006
# Deviatia standard pentru aproximări: 0.004491781383816451