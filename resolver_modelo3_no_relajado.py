import pandas as pd
import json
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, LpMaximize

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
            lpSum(Bik[i][k] * y[k] for k in planificaciones_idx) >= 1,
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

# Conjunto de operaciones
operaciones = operaciones_data["Código operación"].tolist()


# Leer planificaciones del archivo JSON
with open("planificaciones_resultado_antiguo.json", "r") as json_file:
    data = json.load(json_file)

# Extraer planificaciones
planificaciones = data.get("planificaciones", [])

# Adaptar planificaciones al formato esperado por la función
planificaciones_adaptadas = []
for planificacion in planificaciones:
    planificaciones_adaptadas.append([
        {"Código operación": codigo_operacion} for codigo_operacion in planificacion
    ])

# Llamar a la función resolver_maestro con los datos procesados
relajado = False  
rmp, planificaciones_seleccionadas = resolver_maestro(planificaciones_adaptadas, relajado)

# Mostrar los resultados
print("Planificaciones seleccionadas:")
for planificacion in planificaciones_seleccionadas:
    print(planificacion)