import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, LpMaximize

# 1. Leer datos
operaciones_data = pd.read_excel('/Users/almudena/Documents/entrega-3-gc-grupo-mca-a/241204_datos_operaciones_programadas.xlsx')
operaciones_data['Hora inicio '] = pd.to_datetime(operaciones_data['Hora inicio '])
operaciones_data['Hora fin'] = pd.to_datetime(operaciones_data['Hora fin'])

# 2. Generar planificaciones factibles (mínimas)
planificaciones = []  # Comenzamos con una lista vacía para construir combinaciones


# Crear índices para operaciones y planificaciones
operaciones = operaciones_data['Código operación'].tolist()
planificaciones_idx = list(range(len(planificaciones)))
operaciones_idx = {operaciones_data['Código operación'].iloc[i]: i for i in range(len(operaciones_data))}

# 3. Calcular Bik
Bik = {i: {k: 0 for k in planificaciones_idx} for i in operaciones}  # Inicializamos la matriz Bik

# Llenar Bik: Marca que la operación está en la planificación
for k, planificacion in enumerate(planificaciones):
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

    # Obtener precios 
    precios_sombra = {op: rmp.constraints[f"Cubrir_operacion_{op}"].pi for op in operaciones}

    # Generar nueva columna (planificación) basada en los precios sombra
    generacion_columnas = LpProblem("generacion_columnas", LpMaximize)

    quirofanos = range(len(costes))
    costes = pd.read_excel('/Users/almudena/Documents/entrega-3-gc-grupo-mca-a/241204_costes.xlsx', index_col=0)
    x = LpVariable.dicts("Asignación operación i a quirófano j", [i for i in operaciones], cat='Binary')
   
    generacion_columnas += lpSum(operaciones_data.loc[i, 'Duración'] * x[i]  for i in operaciones ) <= 1440 

    overlapping_pairs = []
    for i in operaciones:
        for k in operaciones:
            if i < k: 
                if (operaciones_data.loc[k, 'Hora inicio '] < operaciones_data.loc[i, 'Hora fin']):
                    overlapping_pairs.append((i, k))

    for (i, k) in overlapping_pairs:
        generacion_columnas += x[i] + x[k] <= 1

    generacion_columnas += lpSum(x[i]*precios_sombra[i] for i in operaciones)

    generacion_columnas.solve()
    
    if generacion_columnas.objective.value() <= 1:
        break
    
    nueva_planificacion = []
    for i in operaciones:
        if value(x[i]) == 1:
            nueva_planificacion.append({
                'Código operación': i,
                'Hora inicio ': operaciones_data.loc[i, 'Hora inicio '],
                'Hora fin': operaciones_data.loc[i, 'Hora fin']
            })
    planificaciones.append(nueva_planificacion)

   

    

# 6. Caracterizar la solución
solucion = {
    "Número de quirófanos": sum(value(y[k]) for k in planificaciones_idx),
    "Planificaciones seleccionadas": [planificaciones[k] for k in planificaciones_idx if value(y[k]) == 1]
}

# 7. Mostrar resultados
print(f"Número mínimo de quirófanos: {solucion['Número de quirófanos']}")
for idx, planificacion in enumerate(solucion["Planificaciones seleccionadas"]):
    print(f"Planificación {idx + 1}:")
    for op in planificacion:
        print(f" - Operación: {op['Código operación']}, Inicio: {op['Hora inicio ']}, Fin: {op['Hora fin']}")
