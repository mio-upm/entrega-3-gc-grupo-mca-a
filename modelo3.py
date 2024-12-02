import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import plotly.graph_objects as go
import numpy as np

# Leer datos
costes = pd.read_excel('/Users/almudena/Documents/entrega-3-gc-grupo-mca-a/241204_costes.xlsx', index_col=0)
operaciones_data = pd.read_excel('/Users/almudena/Downloads/241204_datos_operaciones_programadas copia.xlsx')

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
quirofanos = range(len(costes))
operaciones = range(len(operaciones_data))
x = LpVariable.dicts("Asignación operación i a quirófano j", [(i, j) for i in operaciones for j in quirofanos], cat='Binary')
y = LpVariable.dicts("Utilización quirofano j",[j for j in quirofanos], lowBound=0, upBound=1, cat='Binary')

# Función objetivo
model += lpSum(y[j] for j in quirofanos)

# Restricciones 
for i in operaciones:
    model += lpSum(x[i, j] for j in quirofanos) == 1

for j in quirofanos:
    model += lpSum(operaciones_data.loc[i, 'Duración'] * x[i, j] for i in range(1, len(operaciones)) if operaciones_data.loc[i, 'Hora inicio '] >= operaciones_data.loc[i-1, 'Hora fin']) <= 1440 * y[j]

for j in range(2, len(quirofanos)):
    model += y[j-1] >= y[j]

# Resolver el modelo
model.solve()


# Resultados
asignaciones = {j: [] for j in quirofanos}
for j in quirofanos:
    for i in operaciones:
        if x[i, j].value() == 1:
            asignaciones[j].append(i)

for j in quirofanos:
    print(f"Quirófano {j}: Operaciones {asignaciones[j]}")
   

# Configuración del gráfico
fig = go.Figure()

# Configuración del eje X (horario)
hora_inicio = pd.Timestamp('2024-12-04 00:00')
hora_fin = pd.Timestamp('2024-12-04 23:59')
x_ticks = pd.date_range(hora_inicio, hora_fin, freq='15min')

# Generar colores únicos para cada operación
np.random.seed(42)
colores = {
    op: f"rgb({r},{g},{b})"
    for op, (r, g, b) in zip(
        operaciones_data.index,
        zip(
            np.random.randint(0, 256, len(operaciones_data)),
            np.random.randint(0, 256, len(operaciones_data)),
            np.random.randint(0, 256, len(operaciones_data)),
        ),
    )
}

# Dibujar operaciones asignadas
for quir_idx, quir in enumerate(quirofanos):
    for op in asignaciones[quir]:
        op_data = operaciones_data.iloc[op]
        inicio = op_data['Hora inicio ']
        fin = op_data['Hora fin']
        x_inicio = (inicio - hora_inicio).total_seconds() / 60
        x_fin = (fin - hora_inicio).total_seconds() / 60
        fig.add_trace(go.Scatter(
            x=[x_inicio, x_fin],
            y=[quir_idx, quir_idx],
            mode='lines',
            line=dict(color=colores[op], width=10),
            name=f"Op {op}, Quir {quir}",
            hoverinfo="text",
            text=f"Operación: {op}, Quirófano: {quir}"
        ))

# Configurar ejes
fig.update_yaxes(
    tickvals=[i for i in range(len(quirofanos))],
    ticktext=[f"Quirófano {j+1}" for j in quirofanos],
    title_text="Quirófanos"
)

fig.update_xaxes(
    tickvals=[(t - hora_inicio).total_seconds() / 60 for t in x_ticks],
    ticktext=[t.strftime('%H:%M') for t in x_ticks],
    title_text="Horario",
    tickangle=45
)

# Configurar diseño del gráfico
fig.update_layout(
    title="Asignación de Operaciones a Quirófanos",
    height=600,
    width=1200,
    margin=dict(l=50, r=50, t=50, b=100)
)

# Mostrar gráfico
fig.show()