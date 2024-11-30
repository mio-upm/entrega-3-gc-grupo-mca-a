import itertools
import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import plotly.graph_objects as go
import numpy as np

# Leer datos
costes = pd.read_excel('/Users/almudena/Documents/entrega-3-gc-grupo-mca-a/241204_costes.xlsx', index_col=0)
operaciones_data = pd.read_excel('/Users/almudena/Documents/entrega-3-gc-grupo-mca-a/241204_datos_operaciones_programadas.xlsx')

# Ordenar las operaciones por hora de inicio de menos a mayor
operaciones_data['Hora inicio '] = pd.to_datetime(operaciones_data['Hora inicio '])
operaciones_data['Hora fin'] = pd.to_datetime(operaciones_data['Hora fin'])
operaciones_data = operaciones_data.sort_values(by='Hora inicio ')

# Calcukar el tiempo de duración de cada operación
operaciones_data['Duración'] = (operaciones_data['Hora fin'] - operaciones_data['Hora inicio ']).dt.total_seconds() / 60
print(operaciones_data['Duración'])

# Modelo
model = LpProblem("Minimización_de_quirofanos", LpMinimize)

# Variables
quirofanos = costes.index
operaciones = operaciones_data['Código operación']
x = LpVariable.dicts("Asignación operación i a quirófano j", [(i, j) for i in operaciones for j in quirofanos], cat='Binary')
y = LpVariable.dicts("Utilización quirofano j", quirofanos, lowBound=0, upBound=1, cat='Binary')

# Función objetivo
model += lpSum(y[j] for j in quirofanos)

# Restricciones 
for i in operaciones:
    model += lpSum(x[i, j] for j in quirofanos) == 1

for j in quirofanos:
    model += lpSum(operaciones_data.loc[i, 'Duración'] * x[i, j] for i in range(1, len(operaciones)) if operaciones_data.loc[i, 'Hora inicio '] >= operaciones_data.loc[i-1, 'Hora fin']) <= 1440 * y[j]

# Resolver el modelo
model.solve()
   
