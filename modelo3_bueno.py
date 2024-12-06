import pandas as pd
import json
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, LpMaximize

# 1. Leer datos
operaciones_data = pd.read_excel("241204_datos_operaciones_programadas.xlsx")
operaciones_data["Hora inicio "] = pd.to_datetime(operaciones_data["Hora inicio "])
operaciones_data["Hora fin"] = pd.to_datetime(operaciones_data["Hora fin"])

# Calcular la duración en minutos
operaciones_data["Duración"] = (
    operaciones_data["Hora fin"] - operaciones_data["Hora inicio "]
).dt.total_seconds() / 60

# Reemplazar espacios y guiones en "Código Operación" por barra baja
operaciones_data["Código operación"] = operaciones_data["Código operación"].str.replace(
    r"[ \-]", "_", regex=True
)

# # Filtrar por especialidades indicadas
# especialidades = [
#     "Cardiología Pediátrica",
#     "Cirugía Cardíaca Pediátrica",
# ]
# operaciones_data = operaciones_data[
#     operaciones_data["Especialidad quirúrgica"].isin(especialidades)
# ]

# Conjunto de operaciones
operaciones = operaciones_data["Código operación"].tolist()


# 2. Generar planificaciones factibles
def generar_planificaciones_por_operacion(operaciones):
    print("INICIO")
    """Genera una planificación para cada operación."""
    planificaciones = []

    for op in operaciones:
        # Crear una planificación que contenga únicamente la operación actual
        planificaciones.append([op])

    print("FINAL")
    return planificaciones


planificaciones = generar_planificaciones_por_operacion(
    operaciones_data.to_dict("records")
)


def resolver_maestro(planificaciones, relajado):
    # Crear índices para operaciones y planificaciones
    planificaciones_idx = list(range(len(planificaciones)))

    # 3. Calcular Bik
    Bik = {
        i: {k: 0 for k in planificaciones_idx} for i in operaciones
    }  # Inicializamos la matriz Bik

    # Llenar Bik: Marca que la operación está en la planificación
    for k, planificacion in enumerate(planificaciones):
        for operacion in planificacion:
            codigo = operacion["Código operación"]
            Bik[codigo][k] = 1  # Marcar que la operación está en la planificación k

    # 4. Construir el modelo maestro inicial (RMP)
    rmp = LpProblem("Minimizar_quirófanos", LpMinimize)

    # Variables: y_k indica si se selecciona la planificación k
    if relajado:
        y = LpVariable.dicts("y", planificaciones_idx, lowBound=0, cat="Continuous")
    else:
        y = LpVariable.dicts("y", planificaciones_idx, cat="Binary")

    # Restricción de cobertura: cada operación debe estar cubierta por al menos una planificación
    for i in operaciones:
        rmp += (
            lpSum(Bik[i][k] * y[k] for k in planificaciones_idx) == 1,
            f"Cubrir_operacion_{i}",
        )

    # Función objetivo: minimizar el número de quirófanos seleccionados (sin considerar el coste)
    rmp += lpSum(y[k] for k in planificaciones_idx), "Minimizar_numero_quirófanos"

    # Resolver el RMP actual
    rmp.solve()

    print("Variables seleccionadas:")
    for variable in rmp.variables():
        if variable.varValue > 0:
            print(f"{variable.name} = {variable.varValue}")

    # Filtrar y retornar solo las planificaciones asociadas a las variables y_k con valor 1
    planificaciones_seleccionadas = [
        planificaciones[k] for k in planificaciones_idx if value(y[k]) > 0
    ]

    # Guardar planificaciones seleccionadas en un archivo JSON

    codigos_planificaciones_seleccionadas = [
        [operacion["Código operación"] for operacion in planificacion]
        for planificacion in planificaciones_seleccionadas
    ]

    codigos_planificaciones = [
        [operacion["Código operación"] for operacion in planificacion]
        for planificacion in planificaciones
    ]

    resultado = {
        "planificaciones_seleccionadas": codigos_planificaciones_seleccionadas,
        "planificaciones": codigos_planificaciones,
    }

    with open("planificaciones_resultado.json", "w") as json_file:
        json.dump(resultado, json_file, indent=4)

    return rmp, planificaciones_seleccionadas


# 5. Generar nuevas columnas (problema subyacente)
while True:

    rmp, planificaciones_seleccionadas = resolver_maestro(planificaciones, True)

    print(
        [{"name": name, "precio sombra": c.pi} for name, c in rmp.constraints.items()]
    )

    # Aplicar la función al acceder a los precios sombra
    precios_sombra = {
        op: rmp.constraints[f"Cubrir_operacion_{op}"].pi for op in operaciones
    }

    # Generar nueva columna (planificación) basada en los precios sombra
    generacion_columnas = LpProblem("generacion_columnas", LpMaximize)

    x = LpVariable.dicts(
        "Asignación operación i a quirófano j", [i for i in operaciones], cat="Binary"
    )

    generacion_columnas += (
        lpSum(
            operaciones_data.loc[
                operaciones_data["Código operación"] == i, "Duración"
            ].values[0]
            * x[i]
            for i in operaciones
        )
        <= 1440
    )

    overlapping_pairs = []
    for i in operaciones:
        for k in operaciones:
            if i < k:
                if (
                    operaciones_data.loc[
                        operaciones_data["Código operación"] == i, "Hora inicio "
                    ].values[0]
                    < operaciones_data.loc[
                        operaciones_data["Código operación"] == k, "Hora fin"
                    ].values[0]
                ) and (
                    operaciones_data.loc[
                        operaciones_data["Código operación"] == i, "Hora fin"
                    ].values[0]
                    > operaciones_data.loc[
                        operaciones_data["Código operación"] == k, "Hora inicio "
                    ].values[0]
                ):
                    overlapping_pairs.append((i, k))

    for i, k in overlapping_pairs:
        generacion_columnas += x[i] + x[k] <= 1

    generacion_columnas += lpSum(x[i] * precios_sombra[i] for i in operaciones)

    generacion_columnas.solve()

    if generacion_columnas.objective.value() <= 1:
        break

    nueva_planificacion = []
    for i in operaciones:
        if value(x[i]) == 1:
            nueva_planificacion.append(
                {
                    "Código operación": i,
                    "Hora inicio ": operaciones_data.loc[
                        operaciones_data["Código operación"] == i, "Hora inicio "
                    ].values[0],
                    "Hora fin": operaciones_data.loc[
                        operaciones_data["Código operación"] == k, "Hora fin"
                    ].values[0],
                }
            )
    planificaciones.append(nueva_planificacion)


planificaciones_idx = list(range(len(planificaciones)))

# 6. Caracterizar la solución
rmp, planificaciones_finales = resolver_maestro(planificaciones, False)
solucion = {
    "Número de quirófanos": len(planificaciones_finales),
    "Planificaciones seleccionadas": planificaciones_finales,
}

# 7. Mostrar resultados
print(f"Número mínimo de quirófanos: {solucion['Número de quirófanos']}")
for idx, planificacion in enumerate(solucion["Planificaciones seleccionadas"]):
    print(f"Planificación {idx + 1}:")
    for op in planificacion:
        print(
            f" - Operación: {op['Código operación']}, Inicio: {op['Hora inicio ']}, Fin: {op['Hora fin']}"
        )
