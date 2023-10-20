from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarma'), ('Incendiu', 'Alarma')])


# Probabilitatea unui cutremur
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])

# Probabilitatea unui incendiu în funcție de cutremur
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2,
                          values=[[0.0099, 0.03],
                                  [0.9901, 0.97]],
                          evidence=['Cutremur'],
                          evidence_card=[2])




# Probabilitatea declanșării alarmei în funcție de cutremur și incendiu
cpd_alarma = TabularCPD(variable='Alarma', variable_card=2,
                        values=[[0.9799, 0.98, 0.01, 0.02],
                                [0.0201, 0.02, 0.99, 0.98]],
                        evidence=['Cutremur', 'Incendiu'],
                        evidence_card=[2, 2])

# Adăugarea CPD-urilor la model
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

# Verificarea modelului
assert model.check_model()

# Realizarea inferenței folosind Eliminarea Variabilelor
infer = VariableElimination(model)

# Interogați modelul pentru probabilitatea unui incendiu dat un cutremur
result = infer.query(variables=['Incendiu'], evidence={'Cutremur': 1})
print(result)


# Calcularea probabilității marginale a alarmei de incendiu
result_alarm = infer.query(variables=['Alarma'])

# Calcularea probabilității condiționate P(Alarma = 1 | Cutremur = 1)
result_alarm_given_earthquake = infer.query(variables=['Alarma'], evidence={'Cutremur': 1})

# Calcularea probabilității P(Cutremur = 1 | Alarma = 1) folosind teorema lui Bayes
prob_cutremur_given_alarm = (result_alarm_given_earthquake.values[1] * cpd_cutremur.values[1]) / result_alarm.values[1]

print(prob_cutremur_given_alarm)


# # Calcularea probabilității că alarma de incendiu nu este activată
# result_no_alarm = infer.query(variables=['Alarma'], evidence={'Alarma': 0})
#
# # Calcularea probabilității că un incendiu a avut loc fără ca alarma de incendiu să se activeze
# result_fire_given_no_alarm = infer.query(variables=['Incendiu'])
#
# prob_fire_given_no_alarm = result_fire_given_no_alarm.values[1]
#
# print(prob_fire_given_no_alarm)


