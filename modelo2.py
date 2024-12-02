import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import matplotlib.pyplot as plt


# 1. Leer datos de operaciones quirúrgicas
operaciones_data = pd.read_excel('241204_datos_operaciones_programadas.xlsx')

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
costes_data = pd.read_excel('241204_costes.xlsx', index_col=0)
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
