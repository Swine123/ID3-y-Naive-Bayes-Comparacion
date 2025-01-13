import pandas as pd
import numpy as np

# Función para cargar y preprocesar los datos
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: El archivo '{file_path}' no se encontró.")
        return None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

    # Manejo de valores nulos
    data = data.fillna("Desconocido")
    
    # Conversión de datos categóricos a numéricos
    for column in data.columns:
        if data[column].dtype == 'object':
            unique_vals = data[column].unique()
            val_dict = {val: idx for idx, val in enumerate(unique_vals)}
            data[column] = data[column].map(val_dict).fillna(-1).astype(int)
    
    return data

# Función para calcular las probabilidades
def calculate_probabilities(data, features, target):
    # Probabilidad de cada clase
    class_counts = data[target].value_counts().to_dict()
    total_count = len(data)
    class_probabilities = {cls: count / total_count for cls, count in class_counts.items()}
    
    # Probabilidad condicional de cada característica dado la clase
    conditional_probabilities = {}
    for feature in features:
        conditional_probabilities[feature] = {}
        for cls in class_counts:
            feature_class_counts = data[data[target] == cls][feature].value_counts().to_dict()
            total_class_count = class_counts[cls]
            conditional_probabilities[feature][cls] = {val: count / total_class_count for val, count in feature_class_counts.items()}
    
    return class_probabilities, conditional_probabilities

# Función para predecir la clase usando Naive Bayes
def predict_naive_bayes(sample, class_probabilities, conditional_probabilities, features):
    max_class = None
    max_prob = -1
    
    for cls in class_probabilities:
        prob = class_probabilities[cls]
        for feature in features:
            feature_val = sample.get(feature, -1)  # Usa un valor por defecto si falta el feature
            prob *= conditional_probabilities[feature][cls].get(feature_val, 1e-6)  # Valor pequeño para evitar probabilidad cero
    
        if prob > max_prob:
            max_class = cls
            max_prob = prob
            
    return max_class

# Ruta del archivo
file_path = r"C:\Users\e_dso\Downloads\Datos filtrados.xlsx"

# Carga de datos
data = load_and_preprocess_data(file_path)
if data is not None:
    # Definición de características y objetivo
    features = ['MES', 'ID_DIA', 'DIASEMANA', 'TIPACCID', 'ZONA']
    target = 'AUTOMOVIL'
    
    # Cálculo de probabilidades
    class_probabilities, conditional_probabilities = calculate_probabilities(data, features, target)
    
    # Prueba de predicción con un ejemplo
    sample = data.iloc[0].to_dict()  # Convertimos la fila a diccionario
    predicted_class = predict_naive_bayes(sample, class_probabilities, conditional_probabilities, features)
    
    print(f"Predicted class for the sample: {predicted_class}")
