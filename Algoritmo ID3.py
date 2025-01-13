import pandas as pd
import numpy as np
from graphviz import Digraph

# Función para calcular la entropía
def calcular_entropia(y):
    proporciones = np.bincount(y) / len(y)
    return -np.sum([p * np.log2(p) for p in proporciones if p > 0])

# Función para calcular la ganancia de información para una característica
def calcular_ganancia_informacion(datos, caracteristica, objetivo):
    datos = datos.sort_values(by=caracteristica).reset_index(drop=True)
    umbrales = (datos[caracteristica][:-1] + datos[caracteristica][1:]) / 2
    entropia_antes = calcular_entropia(datos[objetivo].values)

    tabla_ganancia_info = []
    for umbral in umbrales:
        division_izquierda = datos[datos[caracteristica] <= umbral][objetivo]
        division_derecha = datos[datos[caracteristica] > umbral][objetivo]

        if len(division_izquierda) > 0 and len(division_derecha) > 0:
            entropia_ponderada_despues = (
                len(division_izquierda) / len(datos) * calcular_entropia(division_izquierda.values) +
                len(division_derecha) / len(datos) * calcular_entropia(division_derecha.values)
            )
            ganancia = entropia_antes - entropia_ponderada_despues
            tabla_ganancia_info.append([umbral, len(division_izquierda), len(division_derecha), entropia_ponderada_despues, ganancia])

    columnas = ['Umbral', 'Conteo Izquierda', 'Conteo Derecha', 'Entropía Después de la División', 'Ganancia de Información']
    return pd.DataFrame(tabla_ganancia_info, columns=columnas)

# Función recursiva para construir el árbol de decisión
def construir_arbol(datos, caracteristicas, objetivo, profundidad=0, profundidad_max=3):
    if len(np.unique(datos[objetivo])) == 1 or profundidad == profundidad_max:
        clase_mayoritaria = datos[objetivo].value_counts().idxmax()
        return {'clase': clase_mayoritaria}

    resultados_ganancia_info = {}
    for caracteristica in caracteristicas:
        tabla_ganancia = calcular_ganancia_informacion(datos, caracteristica, objetivo)
        if not tabla_ganancia.empty:
            resultados_ganancia_info[caracteristica] = tabla_ganancia

    if not resultados_ganancia_info:
        clase_mayoritaria = datos[objetivo].value_counts().idxmax()
        return {'clase': clase_mayoritaria}

    mejor_caracteristica = max(
        resultados_ganancia_info,
        key=lambda c: resultados_ganancia_info[c]['Ganancia de Información'].max()
    )
    mejor_tabla_ganancia = resultados_ganancia_info[mejor_caracteristica]

    if mejor_tabla_ganancia['Ganancia de Información'].max() <= 0:
        clase_mayoritaria = datos[objetivo].value_counts().idxmax()
        return {'clase': clase_mayoritaria}

    mejor_umbral = mejor_tabla_ganancia.loc[mejor_tabla_ganancia['Ganancia de Información'].idxmax(), 'Umbral']

    print(f"\nProfundidad del Nodo: {profundidad}, Característica: {mejor_caracteristica}")
    print(mejor_tabla_ganancia)

    datos_izquierda = datos[datos[mejor_caracteristica] <= mejor_umbral]
    datos_derecha = datos[datos[mejor_caracteristica] > mejor_umbral]

    return {
        'caracteristica': mejor_caracteristica,
        'umbral': mejor_umbral,
        'izquierda': construir_arbol(datos_izquierda, caracteristicas, objetivo, profundidad + 1, profundidad_max),
        'derecha': construir_arbol(datos_derecha, caracteristicas, objetivo, profundidad + 1, profundidad_max)
    }

# Función para visualizar el árbol de decisión usando graphviz
def visualizar_arbol(arbol, dot=None, padre=None, etiqueta_borde=None):
    if dot is None:
        dot = Digraph()
        dot.attr('node', shape='ellipse')

    if 'clase' in arbol:
        etiqueta = f"Clase: {arbol['clase']}"
        dot.node(str(id(arbol)), label=etiqueta)
        if padre:
            dot.edge(padre, str(id(arbol)), label=etiqueta_borde)
    else:
        etiqueta = f"{arbol['caracteristica']} <= {arbol['umbral']}"
        dot.node(str(id(arbol)), label=etiqueta)
        if padre:
            dot.edge(padre, str(id(arbol)), label=etiqueta_borde)

        visualizar_arbol(arbol['izquierda'], dot, str(id(arbol)), 'Verdadero')
        visualizar_arbol(arbol['derecha'], dot, str(id(arbol)), 'Falso')

    return dot

# Cargar el conjunto de datos
ruta_archivo = 'C:/Users/irvin/OneDrive/Documents/Codigos IIA/Tercer semestre/Estructura de datos/Datos filtrados.xlsx'
datos = pd.read_excel(ruta_archivo)

# Codificar variables categóricas manualmente
for columna in datos.columns:
    if datos[columna].dtype == 'object':
        valores_unicos = datos[columna].unique()
        dic_valores = {val: idx for idx, val in enumerate(valores_unicos)}
        datos[columna] = datos[columna].map(dic_valores)

# Definir características y objetivo
caracteristicas = ['MES', 'ID_DIA', 'DIASEMANA', 'TIPACCID', 'ZONA']
objetivo = 'AUTOMOVIL'

# Construir el árbol de decisión
print("Construyendo el árbol de decisión...")
arbol = construir_arbol(datos, caracteristicas, objetivo, profundidad_max=4)
print("\nConstrucción del árbol de decisión completa.")

# Visualizar el árbol de decisión
print("Visualizando el árbol de decisión...")
dot = visualizar_arbol(arbol)
dot.render('arbol_decision', format='png', cleanup=True)  # Guarda el árbol como un archivo PNG
dot.view()  # Abre la imagen renderizada del árbol
