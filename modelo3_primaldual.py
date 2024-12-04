# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:46:04 2024

@author: Inés Bauret
"""

import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import plotly.graph_objects as go
import numpy as np

# Leer datos
costes = pd.read_excel('C:/Users/TESTER/Desktop/241204_costes.xlsx', index_col=0)
operaciones_data = pd.read_excel('C:/Users/TESTER/Desktop/241204_datos_operaciones_programadas.xlsx')

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

# Crear gráfico
fig = go.Figure()

# Configuración del eje X
hora_inicio = pd.Timestamp('2024-12-04 00:00')
hora_fin = pd.Timestamp('2024-12-04 23:59')
x_ticks = pd.date_range(hora_inicio, hora_fin, freq='15min')

# Generar colores únicos para cada operación
np.random.seed(42)
colores = {
    op: f"rgb({r},{g},{b})"
    for op, (r, g, b) in zip(
        operaciones_data["Código operación"],
        zip(
            np.random.randint(0, 256, len(operaciones_data)),
            np.random.randint(0, 256, len(operaciones_data)),
            np.random.randint(0, 256, len(operaciones_data)),
        ),
    )
}

# Dibujar operaciones
for quir_idx, quir in enumerate(quirofanos):
    for op in asignaciones[quir]:
        op_data = operaciones_data[operaciones_data['Código operación'] == op]
        if not op_data.empty:
            inicio = op_data['Hora inicio '].iloc[0]
            fin = op_data['Hora fin'].iloc[0]
            x_inicio = (inicio - hora_inicio).total_seconds() / 60
            x_fin = (fin - hora_inicio).total_seconds() / 60
            fig.add_trace(go.Scatter(
                x=[x_inicio, x_fin],
                y=[quir_idx, quir_idx],
                mode='lines',
                line=dict(color=colores[op], width=10),
                name=f"{op} ({quir})",
                hoverinfo="text",
                text=[f"Operación: {op}, Quirófano: {quir}"]
            ))

# Configurar ejes
fig.update_yaxes(
    tickvals=[i for i in range(len(quirofanos))],
    ticktext=quirofanos,
    title_text="Quirófanos",
    tickangle=0
)

fig.update_xaxes(
    tickvals=[(t - hora_inicio).total_seconds() / 60 for t in x_ticks],  # Convertir las horas a minutos
    ticktext=[t.strftime('%H:%M') for t in x_ticks],
    title_text="Horario (cada 15 minutos)",
    tickangle=45
)

fig.update_layout(
    title="Asignación de Operaciones a Quirófanos",
    xaxis_title="Horario",
    yaxis_title="Quirófanos",
    height=2000,
    width=2500,
    margin=dict(l=50, r=50, t=50, b=100),
)

# Mostrar gráfico
fig.show()

fig.write_html("solucion_modelo1.html")
import webbrowser
webbrowser.open("solucion_modelo1.html")


#%%
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import matplotlib.pyplot as plt


# 1. Leer datos de operaciones quirúrgicas
operaciones_data = pd.read_excel('C:/Users/TESTER/Desktop/241204_datos_operaciones_programadas.xlsx')

# Filtrar por especialidades indicadas
especialidades = [
    "Cardiología Pediátrica",
    "Cirugía Cardíaca Pediátrica",
    "Cirugía Cardiovascular",
    "Cirugía General y del Aparato Digestivo"
]
operaciones_data = operaciones_data[operaciones_data['Especialidad quirúrgica'].isin(especialidades)]

# Convertir las columnas de hora a formato datetime
operaciones_data['Hora inicio '] = pd.to_datetime(operaciones_data['Hora inicio '])
operaciones_data['Hora fin'] = pd.to_datetime(operaciones_data['Hora fin'])

# 2. Leer y calcular costes promedio de las operaciones
costes_data = pd.read_excel('C:/Users/TESTER/Desktop/241204_costes.xlsx', index_col=0)
quirofanos = costes_data.index.tolist()

# Calcular el coste promedio por operación
coste_medio_operaciones = costes_data.mean(axis=0).to_dict()

# Añadir el coste medio a las operaciones
operaciones_data['Coste'] = operaciones_data['Código operación'].map(coste_medio_operaciones)

# 3. Generar planificaciones factibles (mínimas)
def generar_planificaciones_incremental(operaciones):
    """Genera combinaciones factibles de forma incremental."""
    planificaciones = [[]]

    for op in operaciones:
        nuevas_planificaciones = []
        for planificacion in planificaciones:
            # Verificar si se puede añadir la operación sin solapamiento
            if all(op['Hora inicio '] >= p['Hora fin'] or op['Hora fin'] <= p['Hora inicio '] for p in planificacion):
                nuevas_planificaciones.append(planificacion + [op])
        planificaciones.extend(nuevas_planificaciones)
    return planificaciones

# Generar planificaciones para el conjunto de datos
planificaciones_factibles = generar_planificaciones_incremental(operaciones_data.to_dict('records'))

# Crear índices para operaciones y planificaciones
operaciones = operaciones_data['Código operación'].tolist()
planificaciones_idx = list(range(len(planificaciones_factibles)))

# Calcular Bik y Ck
Bik = {i: {k: 0 for k in planificaciones_idx} for i in operaciones}
Ck = {}

for k, planificacion in enumerate(planificaciones_factibles):
    coste_promedio = 0
    for operacion in planificacion:
        codigo = operacion['Código operación']
        Bik[codigo][k] = 1
        coste_promedio += coste_medio_operaciones[codigo]
    Ck[k] = coste_promedio

# 4. Construcción del modelo
model = LpProblem("Set_Covering_Modelo", LpMinimize)

# Variables: y_k indica si se selecciona la planificación k
y = LpVariable.dicts("y", planificaciones_idx, cat='Binary')

# Función objetivo: minimizar el coste total
model += lpSum(Ck[k] * y[k] for k in planificaciones_idx), "Minimizar_coste"

# Restricciones: cada operación debe estar cubierta por al menos una planificación
for i in operaciones:
    model += lpSum(Bik[i][k] * y[k] for k in planificaciones_idx) >= 1, f"Cubrir_operacion_{i}"

# Resolver el modelo
model.solve()

# 5. Caracterizar la solución
solucion = {
    "Planificaciones seleccionadas": [k for k in planificaciones_idx if y[k].value() == 1],
    "Coste total": model.objective.value()
}

# Mostrar resultados
print("Solución:")
for k in solucion["Planificaciones seleccionadas"]:
    print(f"Planificación {k}:")
    for op in planificaciones_factibles[k]:
        print(f" - Operación: {op['Código operación']}, Inicio: {op['Hora inicio ']}, Fin: {op['Hora fin']}")
print(f"Coste total: {solucion['Coste total']}")

print("Solución:")
asignacion_quirofanos = {}  # Guardar asignaciones de quirófanos


# Asignar planificaciones seleccionadas a quirófanos rotativos
quirofanos_asignados = {quirofano: [] for quirofano in quirofanos}
quirofano_actual = 0

for k in solucion["Planificaciones seleccionadas"]:
    quirofano = quirofanos[quirofano_actual % len(quirofanos)]
    quirofanos_asignados[quirofano].append(planificaciones_factibles[k])
    quirofano_actual += 1

# Mostrar asignaciones
for quirofano, planificaciones in quirofanos_asignados.items():
    print(f"{quirofano}:")
    for planificacion in planificaciones:
        for op in planificacion:
            print(f" - Operación: {op['Código operación']}, Inicio: {op['Hora inicio ']}, Fin: {op['Hora fin']}")
print(f"Coste total: {solucion['Coste total']}")

# 6. Ploteo de las planificaciones en los quirófanos
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.tab10.colors

for idx, (quirofano, planificaciones) in enumerate(quirofanos_asignados.items()):
    for planificacion in planificaciones:
        for op in planificacion:
            ax.barh(
                quirofano,
                (op['Hora fin'] - op['Hora inicio ']).total_seconds() / 3600,  # Duración en horas
                left=(op['Hora inicio '] - pd.Timestamp('2024-12-04 00:00:00')).total_seconds() / 3600,  # Inicio relativo
                color=colors[idx % len(colors)],
                edgecolor="black"
            )

# Formato del eje x para horas
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 2)])
ax.set_xlabel("Hora del día")
ax.set_ylabel("Quirófano")
ax.set_title("Planificaciones de Quirófanos")

# Leyenda
legend_items = []
for quirofano, planificaciones in quirofanos_asignados.items():
    label = f"{quirofano}: " + ", ".join(
        [f"{op['Código operación']}" for planificacion in planificaciones for op in planificacion]
    )
    legend_items.append(label)

plt.legend(legend_items, loc="upper left", bbox_to_anchor=(1, 1), title="Operaciones por Quirófano")
plt.tight_layout()
plt.show()


#%%
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value

# 1. Leer datos
operaciones_data = pd.read_excel('C:/Users/TESTER/Desktop/241204_datos_operaciones_programadas.xlsx')
operaciones_data['Hora inicio '] = pd.to_datetime(operaciones_data['Hora inicio '])
operaciones_data['Hora fin'] = pd.to_datetime(operaciones_data['Hora fin'])

# 2. Generar planificaciones factibles (mínimas)
def generar_planificaciones_incremental(operaciones):
    """Genera combinaciones factibles de forma incremental."""
    planificaciones = [[]]  # Comenzamos con una lista vacía para construir combinaciones

    for op in operaciones:
        nuevas_planificaciones = []
        for planificacion in planificaciones:
            # Verificar si se puede añadir la operación sin solapamiento
            if all(op['Hora inicio '] >= p['Hora fin'] or op['Hora fin'] <= p['Hora inicio '] for p in planificacion):
                nuevas_planificaciones.append(planificacion + [op])
        planificaciones.extend(nuevas_planificaciones)
    return planificaciones

# Generar planificaciones para el conjunto de datos
planificaciones_factibles = generar_planificaciones_incremental(operaciones_data.to_dict('records'))

# Crear índices para operaciones y planificaciones
operaciones = operaciones_data['Código operación'].tolist()
planificaciones_idx = list(range(len(planificaciones_factibles)))
operaciones_idx = {operaciones_data['Código operación'].iloc[i]: i for i in range(len(operaciones_data))}

# 3. Calcular Bik y Ck
Bik = {i: {k: 0 for k in planificaciones_idx} for i in operaciones}  # Inicializamos la matriz Bik
Ck = {}

# Se elimina el cálculo de coste ya que no es relevante para la optimización en este caso

# Llenar Bik: Marca que la operación está en la planificación
for k, planificacion in enumerate(planificaciones_factibles):
    for operacion in planificacion:
        codigo = operacion['Código operación']
        Bik[codigo][k] = 1  # Marcar que la operación está en la planificación k

# 4. Construir el modelo maestro inicial (RMP)
rmp = LpProblem("Minimizar_quirófanos", LpMinimize)

# Variables: y_k indica si se selecciona la planificación k
y = LpVariable.dicts("y", planificaciones_idx, cat='Binary')

# Restricción de cobertura: cada operación debe estar cubierta por al menos una planificación
for i in operaciones:
    rmp += lpSum(Bik[i][k] * y[k] for k in planificaciones_idx) >= 1, f"Cubrir_operacion_{i}"

# Función objetivo: minimizar el número de quirófanos seleccionados (sin considerar el coste)
rmp += lpSum(y[k] for k in planificaciones_idx), "Minimizar_numero_quirófanos"

# 5. Generar nuevas columnas (problema subyacente)
while True:
    # Resolver el RMP actual
    rmp.solve()

    # Obtener precios duales
    precios_duales = {op: rmp.constraints[f"Cubrir_operacion_{op}"].pi for op in operaciones}

    # Generar nueva columna (planificación) basada en los precios duales
    nueva_planificacion = []
    operaciones_ordenadas = sorted(
        operaciones_data.to_dict('records'),
        key=lambda op: precios_duales[op['Código operación']], reverse=True
    )

    for op in operaciones_ordenadas:
        if all(op['Hora inicio '] >= p['Hora fin'] or op['Hora fin'] <= p['Hora inicio '] for p in nueva_planificacion):
            nueva_planificacion.append(op)

    # Revisar si la columna es útil (no está vacía)
    if not nueva_planificacion or len(nueva_planificacion) == 0:
        break

    # Añadir columna al modelo
    planificaciones_factibles.append(nueva_planificacion)
    y[len(planificaciones_factibles) - 1] = LpVariable(f"y_{len(planificaciones_factibles) - 1}", cat='Binary')

    # Actualizar restricciones de cobertura
    for op in operaciones:
        if any(op in [p['Código operación'] for p in nueva_planificacion]):
            rmp += y[len(planificaciones_factibles) - 1] >= 1, f"Cobertura_{op}"

# 6. Caracterizar la solución
solucion = {
    "Número de quirófanos": sum(value(y[k]) for k in planificaciones_idx),
    "Planificaciones seleccionadas": [planificaciones_factibles[k] for k in planificaciones_idx if value(y[k]) == 1]
}

# 7. Mostrar resultados
print(f"Número mínimo de quirófanos: {solucion['Número de quirófanos']}")
for idx, planificacion in enumerate(solucion["Planificaciones seleccionadas"]):
    print(f"Planificación {idx + 1}:")
    for op in planificacion:
        print(f" - Operación: {op['Código operación']}, Inicio: {op['Hora inicio ']}, Fin: {op['Hora fin']}")
