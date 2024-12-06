import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt

# Leer datos
costes = pd.read_excel('241204_costes.xlsx', index_col=0)
operaciones_data = pd.read_excel('241204_datos_operaciones_programadas.xlsx')

# Filtrar operaciones de la especialidad "Cardiología Pediátrica"
operaciones_data = operaciones_data[operaciones_data['Especialidad quirúrgica'] == 'Cardiología Pediátrica']

# Procesar datos de incompatibilidades (calcular L_i)
operaciones_data['Hora inicio '] = pd.to_datetime(operaciones_data['Hora inicio '])
operaciones_data['Hora fin'] = pd.to_datetime(operaciones_data['Hora fin'])

L = {}
for i, op_i in operaciones_data.iterrows():
    incompatibles = []
    for j, op_j in operaciones_data.iterrows():
        # No comparar una operación consigo misma
        if i == j:
            continue
        # Comprobar solapamiento de horarios
        if (
            (op_i['Hora inicio '] < op_j['Hora fin']) and
            (op_i['Hora fin'] > op_j['Hora inicio '])
        ):
            incompatibles.append(op_j['Código operación'])
    L[op_i['Código operación']] = incompatibles

# Modelo
model = LpProblem("Minimización_de_costes", LpMinimize)

# Variables
quirofanos = costes.index
operaciones = operaciones_data['Código operación']
x = LpVariable.dicts("Asignación", [(i, j) for i in operaciones for j in quirofanos], cat='Binary')

# Función objetivo
model += lpSum(x[i, j] * costes.loc[j, i] for i in operaciones for j in quirofanos)

# Restricción 1
for i in operaciones:
    model += lpSum(x[i, j] for j in quirofanos) >= 1

# Restricción 2
for i in operaciones:
    for j in quirofanos:
        model += lpSum(x[incompatible_op, j] for incompatible_op in L[i]) + x[i, j] <= 1

# Resolver el modelo
model.solve()

# Resultados
print("Estado del modelo:", model.status)
print("Coste total:", model.objective.value())
asignaciones = {j: [] for j in quirofanos}
for i in operaciones:
    for j in quirofanos:
        if x[i, j].value() == 1:
            print(f"Operación {i} asignada al quirófano {j}")
            asignaciones[j].append(i)

# Graficar los resultados
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.tab10.colors

for idx, (quirofano, ops) in enumerate(asignaciones.items()):
    for op in ops:
        op_data = operaciones_data[operaciones_data['Código operación'] == op]
        if not op_data.empty:
            inicio = (op_data['Hora inicio '].iloc[0] - pd.Timestamp('2024-12-04 00:00')).total_seconds() / 3600
            duracion = (op_data['Hora fin'].iloc[0] - op_data['Hora inicio '].iloc[0]).total_seconds() / 3600
            ax.barh(
                quirofano,
                duracion,
                left=inicio,
                color=colors[idx % len(colors)],
                edgecolor="black"
            )

# Configuración del eje x (horas)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 2)])
ax.set_xlabel("Hora del día")
ax.set_ylabel("Quirófano")
ax.set_title("Asignación de Operaciones a Quirófanos - Modelo 1")

legend_items = []
for quirofano, ops in asignaciones.items():
    if ops:  # Solo incluir quirófanos con operaciones
        label = f"{quirofano}: " + ", ".join(ops)
        legend_items.append(label)

plt.legend(legend_items,loc="upper left",bbox_to_anchor=(1, 1),title="Operaciones por Quirófano")
plt.tight_layout()
plt.show()