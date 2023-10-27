import numpy as np
import scipy.stats

lambda_poisson = 20
media_normal = 2
deviatie_standard_normal = 0.5

alpha_exponential = 5
numar_clienti = np.random.poisson(lambda_poisson)

timp_plasare_plata = np.random.normal(media_normal, deviatie_standard_normal, numar_clienti)
timp_gatire = np.random.exponential(alpha_exponential, numar_clienti)
timp_total_comanda = timp_plasare_plata + timp_gatire

timp_asteptare_client = timp_total_comanda - media_normal

timp_mediu_asteptare = np.mean(timp_asteptare_client)
print("Timpul mediu de așteptare al clienților:", timp_mediu_asteptare, "minute")


probabilitate_dorita = 0.95
timp_maxim = 15

alpha_maxim = -np.log(1 - probabilitate_dorita) / timp_maxim

print("Valoarea maximă a lui α pentru a servi clienții în mai puțin de 15 minute cu o probabilitate de 95%:", alpha_maxim)

lambda_sosire_clienti = 20

timp_mediu_asteptare = 1 / (alpha_maxim - lambda_sosire_clienti)

print("Timpul mediu de așteptare pentru a fi servit unui client:", timp_mediu_asteptare, "minute")


