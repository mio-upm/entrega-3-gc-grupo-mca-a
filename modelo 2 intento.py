#Modelo 2

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum

operaciones_data = pd.read_excel('241204_datos_operaciones_programadas.xlsx')
costes = pd.read_excel('241204_costes.xlsx', index_col=0)

# Filtrar por los cuatro servicios indicados
servicios = ["Cardiología Pediátrica","Cirugía Cardíaca Pediátrica","Cirugía Cardiovascular",
             "Cirugía General y del Aparato Digestivo",]

operaciones_data = operaciones_data[operaciones_data['Especialidad quirúrgica'].isin(servicios)]

# Asegurarnos de que las columnas de hora sean del tipo datetime
operaciones_data['Hora inicio '] = pd.to_datetime(operaciones_data['Hora inicio '])
operaciones_data['Hora fin'] = pd.to_datetime(operaciones_data['Hora fin'])

# Calcular el costo promedio de cada operación
operaciones_data['Costo promedio'] = operaciones_data['Código operación'].map(costes.mean(axis=0).to_dict())

# Función para generar planificaciones sin solapamientos
def generar_planificaciones(operaciones_data):
    planificaciones = []
    operaciones = operaciones_data.sort_values('Hora inicio ')
    
    # Iterar por las operaciones y asignarlas a planificaciones
    while len(operaciones) > 0:
        # Comenzar una nueva planificación
        planificacion = []
        planificacion.append(operaciones.iloc[0])  # Asignamos la primera operación

        # Eliminar la operación de la lista de operaciones disponibles
        operaciones = operaciones.iloc[1:]

        # Asignar más operaciones a la planificación si no hay solapamiento
        for idx, op in operaciones.iterrows():
            if all(op['Hora inicio '] >= op_anterior['Hora fin'] for op_anterior in planificacion):
                planificacion.append(op)
                operaciones = operaciones.drop(idx)

        # Añadir la planificación generada al conjunto de planificaciones
        planificaciones.append(planificacion)

    return planificaciones

# Generar las planificaciones factibles (sin solapamientos)
K = generar_planificaciones(operaciones_data)

# Crear el modelo
model = LpProblem("Modelo_2_Set_Covering", LpMinimize)

# Variables de decisión: y_k indica si la planificación k es seleccionada
y = LpVariable.dicts("Planificación", range(len(K)), cat="Binary")

# Crear la matriz Bik
Bik = {}
for k in range(len(K)):
    Bik[k] = {op['Código operación']: 1 for op in K[k]}
    
# Pre-calcular los valores de Ck
Ck = {}
for k in range(len(K)):
    Ck[k] = sum(Bik[k].get(op, 0) * operaciones_data.loc[operaciones_data['Código operación'] == op, 'Costo promedio'].values[0] for op in Bik[k])

# Crear el modelo
model = LpProblem("Modelo_2_Set_Covering", LpMinimize)

# Variables de decisión: y_k indica si la planificación k es seleccionada
yk = LpVariable.dicts("Planificación", range(len(K)), cat="Binary")

# Función objetivo: minimizar los costos totales de las planificaciones seleccionadas
model += lpSum(
    yk[k] * Ck[k]  # Usamos los valores pre-calculados de Ck
    for k in range(len(K))
)

# Restricciones: cada operación debe estar cubierta por al menos una planificación
for op in operaciones_data.index:
    model += lpSum(yk[k] * Bik[k].get(op, 0) for k in range(len(K))) >= 1

# Resolver el modelo
model.solve()

# Resultados
print("Estado del modelo:", model.status)
print("Costo total:", model.objective.value())

print("Planificaciones seleccionadas:")
for k in range(len(K)):
    if yk[k].value() == 1:
        print(f"Planificación {k}: Operaciones {[op['Código operación'] for op in K[k]]}")