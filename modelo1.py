import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import plotly.graph_objects as go
import numpy as np

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

fig.write_html("output.html")
import webbrowser
webbrowser.open("output.html")
