import arviz as az
import matplotlib.pyplot as plt

import numpy as np

import pymc as pm

start = [0,1]

# 0 => castiga J0  0 = stema
# 1 => castiga J2  

etapa1 = np.random.choice(start) 

if etapa1 == 0:

  jucator1 = 'J0'
  jucator2 = 'J1'

if etapa1 == 1:

  jucator1 = 'J1'
  jucator2 = 'J0'


print(jucator1)
print(jucator2)


#runda1

runda1 = np.random.choice(start)
if runda1 == 0:
  n = 1

print(n)

i = 1
m = 0

while i<= n:
  runda2 = np.random.choice(start)
  if runda2 == 0:
    m = m + 1
  i = i + 1

print(m)

if n >= m:
  print("Castigatorul este: ", jucator1)

else:
  print("Castigatorul este: ", jucator2)

#Pana aici s-a ales random cine incepe jocul, apoi primul jucator arunca moneda, iar al doilea o arunca de n+1 ori
#print:

# J1
# J0
# 1
# 1
# Castigatorul este:  J1

#rezultatul poate fi diferit in functie de ce se alege la etapa1 (cine incepe jocul)




#Simulez jocul de 10.000 ori

prob_stema_j1 = 2/3
start = [0,1]
castig_J0 = 0
castig_J1 = 0


i = 1;

while i <= 10000:

#asociez 2 pt J0 si 3 pt J1

  etapa1 = np.random.choice(start) 

  if etapa1 == 0:
    jucator1 = 2
    jucator2 = 3

  if etapa1 == 1:
    jucator1 = 3
    jucator2 = 2

  if jucator1 == 2: #adica primul jucator este J0
    runda1 = np.random.choice(start) #alegerea va fi normala

  if jucator1 == 3: #adica primul jucator este J1
    runda1 = np.random.choice(start, p = [2/3, 1/3])

  if runda1 == 0:
    n = 1

  j = 1
  m = 0

  while j <= n:
     if jucator1 == 2: #adica primul jucator este J0 deci al doilea J1
        runda2 = np.random.choice(start, p = [2/3, 1/3]) #alegerea va fi facuta de J1, deci e masluita

     if jucator1 == 3: #adica primul jucator este J1, deci al doilea J0
       runda1 = np.random.choice(start) #functioneaza normal
     if runda2 == 0:
      m = m + 1
     j = j + 1

  if n >= m:
    if jucator1 == 2:
      castig_J0 = castig_J0 +1
    if jucator1 == 3:
      castig_J1 = castig_J1 +1


  else:
    if jucator1 == 3:
      castig_J1 = castig_J1 +1
    if jucator1 == 2:
      castig_J0 = castig_J0 +1

  i = i + 1


if castig_J0 > castig_J1:
  print ("Castigatorul J0 are mai multe sanse de castig", castig_J0)

if castig_J0 < castig_J1:
  print ("Castigatorul J1 are mai multe sanse de castig", castig_J1)


if castig_J0 == castig_J1:
  print ("Ambii castigatori au aceeasi sansa de castig", castig_J0, castig_J1)


#Castigatorul J1 are mai multe sanse de castig 5064 asta printeaza la prima rulare


#Retea Bayesiana

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('Start', 'J0'), ('Start', 'J1'), ('J0', 'J1'), ('J1', 'J0')])

#Start ofera posibilitatea jucatorului J0 sau jucatorului J1 sa arunce primul
#Daca J0 este primul jucator care alege primul, al doilea sigur va fi J1
#Daca J1 este primul jucator care alege primul, al doilea sigur va fi J0
