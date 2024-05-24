# BENEMERITA UNIVERSIDAD AUTONOMA DE PUEBLA
# FACULTAD DE CIENCIAS DE LA COMPUTACIÓN

# - - - - - - - - - - - - - - - - - - - - - - - - 

# NOMBRE                                MATRICULA
# Jesus Huerta Aguilar                  202041509
# Jasmine Perez Sanchez                 202073218
# Gabino Ruiz Ramirez                   202056050
# Guadalupe Quetzalli Huitzil Juarez    202128649 

# - - - - - - - - - - - - - - - - - - - - - - - -
# PROYECTO FINAL: RECUPERACIÓN DE LA INFORMACIÓN
#      _____ _______________________  ______       ____  ______   ___    _   _____    __    _________ _________
#     / ___//  _/ ___/_  __/ ____/  |/  /   |     / __ \/ ____/  /   |  / | / /   |  / /   /  _/ ___//  _/ ___/
#     \__ \ / / \__ \ / / / __/ / /|_/ / /| |    / / / / __/    / /| | /  |/ / /| | / /    / / \__ \ / / \__ \ 
#    ___/ // / ___/ // / / /___/ /  / / ___ |   / /_/ / /___   / ___ |/ /|  / ___ |/ /____/ / ___/ // / ___/ / 
#   /____/___//____//_/ /_____/_/  /_/_/  |_|  /_____/_____/  /_/  |_/_/ |_/_/  |_/_____/___//____/___//____/

#                              ____  ____  __    ________________________ 
#                             / __ \/ __ \/ /   /  _/_  __/  _/ ____/ __ \
#                            / /_/ / / / / /    / /  / /  / // /   / / / /
#                           / ____/ /_/ / /____/ /  / / _/ // /___/ /_/ / 
#                          /_/    \____/_____/___/ /_/ /___/\____/\____/  


# █ █ █ █ █ █  ┳┳┳┓┏┓┏┓┳┓┏┳┓┏┓┳┓  ┳┓┳┳┓┓ ┳┏┓┏┳┓┏┓┏┓┏┓┏┓
# █ █ █ █ █ █  ┃┃┃┃┃┃┃┃┣┫ ┃ ┣┫┣┫  ┣┫┃┣┫┃ ┃┃┃ ┃ ┣ ┃ ┣┫┗┓
# █ █ █ █ █ █  ┻┛ ┗┣┛┗┛┛┗ ┻ ┛┗┛┗  ┻┛┻┻┛┗┛┻┗┛ ┻ ┗┛┗┛┛┗┗┛

#    Importa varias bibliotecas necesarias para la aplicación.
#    Estas bibliotecas permiten la manipulación de archivos, visualización de datos,
#    procesamiento de texto, y clasificación de texto.  
import os  
from tkinter import filedialog, messagebox  # Para crear cuadros de diálogo de archivos y mensajes en una GUI.
import matplotlib.pyplot as plt             # Para crear gráficos y visualizaciones.
import seaborn as sns                       # Para crear gráficos estadísticos más atractivos.
from nltk.tokenize import word_tokenize     # Para dividir texto en palabras.
from nltk.corpus import stopwords           # Para acceder a palabras comunes que deben ser ignoradas en el análisis de texto.
from nltk.stem import WordNetLemmatizer     # Para reducir palabras a su forma base (lemmatización).
from collections import Counter             # Para contar la frecuencia de elementos en una lista.
from wordcloud import WordCloud             # Para crear nubes de palabras a partir de texto.
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba.
import unidecode                            # Para eliminar acentos y caracteres especiales de texto.
from datetime import datetime               # Para trabajar con fechas y horas.
import tkinter as tk                        # Para crear interfaces gráficas de usuario.
from tkinter import ttk, simpledialog       # Para widgets avanzados de tkinter y diálogos simples.
from tkinter import simpledialog
import pickle


# █ █ █ █ █ █  ┏┳┓┏┓┳┓┏┓┏┓┳┓┏┓┓ ┏┓┓ ┏
# █ █ █ █ █ █   ┃ ┣ ┃┃┗┓┃┃┣┫┣ ┃ ┃┃┃┃┃
# █ █ █ █ █ █   ┻ ┗┛┛┗┗┛┗┛┛┗┻ ┗┛┗┛┗┻┛

#    Importa varias bibliotecas necesarias para la aplicación.
#    Estas bibliotecas permiten la creación y entrenamiento de modelos de aprendizaje profundo
#    para el procesamiento de texto y clasificación.
import tensorflow as tf                                 # Para construir y entrenar modelos de aprendizaje profundo.
from tensorflow.keras.preprocessing.text import Tokenizer  # Para convertir texto en secuencias de enteros.
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Para ajustar la longitud de las secuencias de texto.
from tensorflow.keras.models import Sequential          # Para construir modelos de aprendizaje profundo de forma secuencial.
from tensorflow.keras.optimizers import Adam            # Para optimizar el modelo durante el entrenamiento.
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D  # Para agregar capas al modelo (embeddings, LSTM, Dense, Dropout).
from sklearn.preprocessing import LabelEncoder          # Para codificar etiquetas en formato numérico.
from tensorflow.keras.callbacks import EarlyStopping    # Para detener el entrenamiento cuando una métrica deja de mejorar.
from tensorflow.keras.layers import Dropout             # Para reducir el sobreajuste en el proceso de entrenamiento.
from tensorflow.keras.models import save_model, load_model # Para guardar y cargar modelos entrenados.
import numpy as np                                      # Para manipulación numérica y operaciones con matrices.


# █ █ █ █ █ █  ┏┓┏┓┳┓┏┓┳┏┓┳┳┳┓┏┓┏┓┳┏┓┳┓  ┳┓┓ ┏┳┓┓┏┓
# █ █ █ █ █ █  ┃ ┃┃┃┃┣ ┃┃┓┃┃┣┫┣┫┃ ┃┃┃┃┃  ┃┃┃  ┃ ┃┫ 
# █ █ █ █ █ █  ┗┛┗┛┛┗┻ ┻┗┛┗┛┛┗┛┗┗┛┻┗┛┛┗  ┛┗┗┛ ┻ ┛┗┛

#    Importa la biblioteca nltk y descarga varios paquetes necesarios.
#    Estos paquetes se utilizan para el procesamiento de texto, como la tokenización,
#    la eliminación de palabras vacías y la lematización.
import nltk                     # Biblioteca para el NLP.
nltk.download('punkt')          # Para la tokenización de texto
nltk.download('stopwords')      # Obtener una lista de palabras vacías
nltk.download('wordnet')        # Lematización (reducción de palabras a su forma base)


# █ █ █ █ █ █  ┏┓┏┓┳┓┏┓┏┓┳┓  ┓┏  ┏┓┳┓┏┓┏┓┳┓┏┓┏┓┏┓┏┓┏┓┳┓  ┏┳┓┏┓┏┓┏┓┏┳┓┏┓┏┓
# █ █ █ █ █ █  ┃ ┣┫┣┫┃┓┣┫┣┫  ┗┫  ┃┃┣┫┣ ┃┃┣┫┃┃┃ ┣ ┗┓┣┫┣┫   ┃ ┣  ┃┃  ┃ ┃┃┗┓
# █ █ █ █ █ █  ┗┛┛┗┛┗┗┛┛┗┛┗  ┗┛  ┣┛┛┗┗┛┣┛┛┗┗┛┗┛┗┛┗┛┛┗┛┗   ┻ ┗┛┗┛┗┛ ┻ ┗┛┗┛

#    Lee archivos de texto (.txt) desde un directorio específico
#    y almacena su contenido en una lista llamada "discursos". Cada elemento de la lista
#    es una tupla que contiene el nombre del archivo y su contenido
def cargar_discursos(directorio):                       
    discursos = []                                          # Crear una lista vacía para almacenar los discursos. 
    for filename in os.listdir(directorio):                 # Iterar sobre los archivos en el directorio especificado.
        if filename.endswith(".txt"):                       # Si el archivo tiene la extensión ".txt".
            with open(os.path.join(directorio, filename), 'r', encoding='utf-8') as file: # Abrir el archivo en modo lectura
                discursos.append((filename, file.read()))   # Leer el contenido del archivo y añadir una tupla a la lista "discursos".
    return discursos                                        # Devolver la lista de discursos

stop_words = set(stopwords.words('spanish'))                # Crear un conjunto de palabras vacías en español utilizando NLTK.
lemmatizer = WordNetLemmatizer()                            # Crear un objeto lematizador para reducir las palabras a su forma base.

#    Esta función preprocesa un texto dado para preparación y análisis posterior.
#    Convierte el texto a minúsculas, elimina acentos, tokeniza el texto,
#    filtra palabras no alfabéticas, palabras vacías y palabras cortas,
#    y reduce las palabras a su forma base
def preprocesar_texto(texto):
    texto = texto.lower()                                   # Convertir el texto a minúsculas.
    texto = unidecode.unidecode(texto)                      # Eliminar acentos y caracteres especiales.
    tokens = word_tokenize(texto)                           # Tokenizar el texto (dividirlo en palabras).
    tokens = [word for word in tokens if word.isalpha()]    # Mantener solo palabras alfabéticas.
    tokens = [word for word in tokens if word not in stop_words]  # Eliminar palabras vacías.
    tokens = [word for word in tokens if len(word) > 3]     # Eliminar palabras menores o iguales a 3 caracteres.
    tokens = [lemmatizer.lemmatize(word) for word in tokens]    # Lemmatizar las palabras.
    return tokens                                           # Devolver la lista de tokens preprocesados.


# █ █ █ █ █ █  ┓┏┏┓┏┓┏┓┳┓┳┳┓ ┏┓┳┓┳┏┓
# █ █ █ █ █ █  ┃┃┃┃┃ ┣┫┣┫┃┃┃ ┣┫┣┫┃┃┃
# █ █ █ █ █ █  ┗┛┗┛┗┛┛┗┻┛┗┛┗┛┛┗┛┗┻┗┛

#    Extraer el vocabulario único de una lista de textos preprocesados.
#    Recorre cada texto, lo tokeniza, y agrega las palabras únicas a un conjunto.
def extraer_vocabulario(textos_preprocesados):
    vocabulario = set()                 # Inicializar un conjunto vacío para almacenar el vocabulario
    for texto in textos_preprocesados:  # Recorrer cada texto preprocesado
        if isinstance(texto, list):     # Verificar si el texto es una lista de palabras
            texto = ' '.join(texto)     # Unir la lista de palabras en un solo string si es una lista
        palabras = word_tokenize(texto) # Tokenizar el texto
        vocabulario.update(palabras)    # Agregar las palabras al conjunto de vocabulario
    return vocabulario                  # Devolver el conjunto de vocabulario

#    Guarda un vocabulario (conjunto de palabras) en un archivo de texto.
#    Escribe cada palabra del vocabulario en una nueva línea en el archivo especificado.
def guardar_vocabulario(vocabulario, archivo_salida):
    with open(archivo_salida, 'w', encoding='utf-8') as file:   # Abrir el archivo de salida en modo escritura con codificación UTF-8
        for palabra in sorted(vocabulario):                     # Recorrer cada palabra en el vocabulario, ordenado alfabéticamente
            file.write(f"{palabra}\n")                          # Escribir cada palabra en una nueva línea


# █ █ █ █ █ █  ┏┓┏┓┏┳┓┏┓┳┓┳┏┓┏┳┓┳┏┓┏┓┏┓
# █ █ █ █ █ █  ┣ ┗┓ ┃ ┣┫┃┃┃┗┓ ┃ ┃┃ ┣┫┗┓
# █ █ █ █ █ █  ┗┛┗┛ ┻ ┛┗┻┛┻┗┛ ┻ ┻┗┛┛┗┗┛

#    Calcula estadísticas descriptivas sobre una lista de textos preprocesados.
#    Devuelve un diccionario con el total de palabras, el promedio de palabras por texto,
#    la longitud máxima y mínima de los textos, y las palabras más frecuentes.
def calcular_estadisticas(textos_preprocesados):
    total_palabras = sum(len(texto) for texto in textos_preprocesados)  # Calcular el total de palabras en todos los textos
    promedio_palabras = total_palabras / len(textos_preprocesados)      # Calcular el promedio de palabras por texto
    longitudes = [len(texto) for texto in textos_preprocesados]         # Obtener las longitudes de cada texto
    longitud_maxima = max(longitudes)                                   # Encontrar la longitud máxima entre los textos
    longitud_minima = min(longitudes)                                   # Encontrar la longitud mínima entre los textos

    palabras_frecuentes = Counter([palabra for texto in textos_preprocesados for palabra in texto]).most_common(10)  # Encontrar las 10 palabras más frecuentes
    return {
        'total_palabras': total_palabras,                               # Devolver el total de palabras
        'promedio_palabras': promedio_palabras,                         # Devolver el promedio de palabras por texto
        'longitud_maxima': longitud_maxima,                             # Devolver la longitud máxima
        'longitud_minima': longitud_minima,                             # Devolver la longitud mínima
        'palabras_frecuentes': palabras_frecuentes                      # Devolver las palabras más frecuentes
    }


# █ █ █ █ █ █  ┳┓┳┏┓┏┓┏┓┳┓┏┓┳┏┓┳┓  ┓ ┏┓┏┓┏┓┳┏┓┏┓
# █ █ █ █ █ █  ┃┃┃┗┓┃┃┣ ┣┫┗┓┃┃┃┃┃  ┃ ┣  ┃┃ ┃┃ ┣┫
# █ █ █ █ █ █  ┻┛┻┗┛┣┛┗┛┛┗┗┛┻┗┛┛┗  ┗┛┗┛┗┛┗┛┻┗┛┛┗

#    Crea una gráfica de dispersión léxica para visualizar la distribución de ciertas palabras
#    a lo largo de una colección de textos preprocesados. La gráfica muestra en qué documentos aparecen
#    las palabras específicas.
def graficar_dispersion_lexica(textos_preprocesados, palabras, label):
    plt.figure(figsize=(10, 6))                 # Crear una figura de tamaño 10x6 pulgadas
    for palabra in palabras:                    # Iterar sobre cada palabra en la lista de palabras
        indices = [i for i, texto in enumerate(textos_preprocesados) if palabra in texto]   # Encontrar los índices de documentos que contienen la palabra
        plt.plot(indices, [palabra] * len(indices), '|', label=palabra)                     # Graficar la dispersión de la palabra en los documentos
    plt.title(f'Dispersión Léxica - {"Humanismo" if label else "Neoliberalismo"}')          # Establecer el título de la gráfica según la etiqueta (label)
    plt.xlabel('Índice del documento')          # Etiquetar el eje x
    plt.ylabel('Palabra')                       # Etiquetar el eje y
    plt.legend()                                # Mostrar la leyenda
    plt.show()                                  # Mostrar la gráfica


# █ █ █ █ █ █  ┏┓┏┓┳┓┳┏┓  ┳┓┏┓  ┏┳┓┳┏┓┳┳┓┏┓┏┓
# █ █ █ █ █ █  ┗┓┣ ┣┫┃┣   ┃┃┣    ┃ ┃┣ ┃┃┃┃┃┃┃
# █ █ █ █ █ █  ┗┛┗┛┛┗┻┗┛  ┻┛┗┛   ┻ ┻┗┛┛ ┗┣┛┗┛

#    Extrae una fecha de un nombre de archivo en formato "dd_mm_yyyy" y la convierte a un objeto de fecha.
#    Si ocurre un error durante el proceso, se imprime un mensaje de error y se devuelve None.
def extraer_fecha(filename):
    try:
        date_str = filename.split('-')[-1].replace('.txt', '')          # Extraer la cadena de fecha del nombre del archivo
        return datetime.strptime(date_str, "%d_%m_%Y")                  # Convertir la cadena de fecha a un objeto datetime
    except Exception as e:                                              # Capturar cualquier excepción que ocurra durante el proceso
        print(f"Error al extraer la fecha del archivo {filename}: {e}") # Imprimir un mensaje de error
        return None                                                     # Devolver None en caso de error

#    Crea una gráfica de series de tiempo para visualizar la frecuencia de ciertas palabras
#    a lo largo del tiempo en una colección de discursos. Extrae las fechas de los nombres de archivo,
#    cuenta las ocurrencias de las palabras y grafica su frecuencia en función del tiempo.
def graficar_series_tiempo(discursos, palabras, titulo):
    series_tiempo = {palabra: [] for palabra in palabras}       # Inicializar un diccionario para almacenar las series de tiempo de cada palabra
    fechas = []                                                 # Inicializar una lista para almacenar las fechas

    for filename, texto in discursos:                           # Iterar sobre cada discurso
        fecha = extraer_fecha(filename)                         # Extraer la fecha del nombre del archivo
        if fecha:                                               # Si se pudo extraer la fecha...
            fechas.append(fecha)                                # Agregar la fecha a la lista de fechas
            tokens = preprocesar_texto(texto)                   # Preprocesar el texto del discurso
            conteo = Counter(tokens)                            # Contar las ocurrencias de cada palabra en el texto
            for palabra in palabras:                            # Iterar sobre cada palabra en la lista de palabras
                series_tiempo[palabra].append(conteo.get(palabra, 0))  # Agregar la frecuencia de la palabra a su serie de tiempo

    # Asegurarse de que las fechas y las series no estén vacías antes de graficar
    if fechas and any(series_tiempo.values()):
        # Ordenar las fechas y las series de tiempo correspondientes
        fechas, series_tiempo_ordenado = zip(*sorted(zip(fechas, zip(*[series_tiempo[palabra] for palabra in palabras]))))

        for i, palabra in enumerate(palabras):                  # Iterar sobre cada palabra en la lista de palabras
            plt.plot(fechas, [serie[i] for serie in series_tiempo_ordenado], label=palabra)  # Graficar la serie de tiempo de la palabra

        plt.xlabel('Fecha')                                     # Etiquetar el eje x
        plt.ylabel('Frecuencia')                                # Etiquetar el eje y
        plt.title(f'Series de Tiempo - {"Humanismo" if titulo else "Neoliberalismo"}')  # Establecer el título de la gráfica según la etiqueta (titulo)
        plt.legend()                                            # Mostrar la leyenda
        plt.show()                                              # Mostrar la gráfica
    else:
        print("No hay datos suficientes para graficar las series de tiempo.")  # Imprimir un mensaje si no hay datos suficientes para graficar


# █ █ █ █ █ █  ┓ ┏┏┓┳┓┳┓┏┓┓ ┏┓┳┳┳┓
# █ █ █ █ █ █  ┃┃┃┃┃┣┫┃┃┃ ┃ ┃┃┃┃┃┃
# █ █ █ █ █ █  ┗┻┛┗┛┛┗┻┛┗┛┗┛┗┛┗┛┻┛

#    Genera una nube de palabras (WordCloud) a partir de una lista de textos preprocesados.
#    Visualiza las palabras más frecuentes en un gráfico de nube de palabras.
def generar_wordcloud(textos_preprocesados, label):
    texto = ' '.join([' '.join(texto) for texto in textos_preprocesados])                   # Unir todos los textos preprocesados en un solo string
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)  # Generar la nube de palabras
    plt.figure(figsize=(10, 6))                                                             # Crear una figura de tamaño 10x6 pulgadas
    plt.imshow(wordcloud, interpolation='bilinear')                                         # Mostrar la nube de palabras con interpolación bilinear
    plt.title(f'WordCloud - {"Humanismo" if label else "Neoliberalismo"}')                  # Establecer el título de la nube de palabras según la etiqueta (label)
    plt.axis('off')                                                                         # Ocultar los ejes
    plt.show()                                                                              # Mostrar la nube de palabras


# █ █ █ █ █ █  ┏┓┳┓┏┓┏┓┳┳┏┓┳┓┏┓┳┏┓  ┳┓┏┓  ┏┓┏┓┓ ┏┓┳┓┳┓┏┓┏┓
# █ █ █ █ █ █  ┣ ┣┫┣ ┃ ┃┃┣ ┃┃┃ ┃┣┫  ┃┃┣   ┃┃┣┫┃ ┣┫┣┫┣┫┣┫┗┓
# █ █ █ █ █ █  ┻ ┛┗┗┛┗┛┗┛┗┛┛┗┗┛┻┛┗  ┻┛┗┛  ┣┛┛┗┗┛┛┗┻┛┛┗┛┗┗┛

#    Crea una gráfica de barras para visualizar la frecuencia de ciertas palabras
#    en una colección de textos preprocesados. Muestra cuántas veces aparece cada palabra
#    en los textos dados.
def graficar_frecuencia_palabras(textos_preprocesados, palabras, label):
    # Contar la frecuencia de cada palabra en la lista de palabras
    frecuencias = Counter([palabra for texto in textos_preprocesados for palabra in texto if palabra in palabras])  
    plt.figure(figsize=(10, 6))                                             # Crear una figura de tamaño 10x6 pulgadas
    sns.barplot(x=list(frecuencias.keys()), y=list(frecuencias.values()))   # Crear una gráfica de barras usando las frecuencias de las palabras
    plt.title(f'Frecuencia de Palabras - {"Humanismo" if label else "Neoliberalismo"}')  # Establecer el título de la gráfica según la etiqueta (label)
    plt.xlabel('Palabra')                       # Etiquetar el eje x
    plt.ylabel('Frecuencia')                    # Etiquetar el eje y
    plt.show()                                  # Mostrar la gráfica


# █ █ █ █ █ █  ┳┓┏┓┏┳┓┏┓┏┓┏┓┳┏┓┳┓  ┳┓┏┓  ┳┓┏┓┏┓┳┳┳┳┓┏┓┳┓┏┳┓┏┓┏┓
# █ █ █ █ █ █  ┃┃┣  ┃ ┣ ┃ ┃ ┃┃┃┃┃  ┃┃┣   ┃┃┃┃┃ ┃┃┃┃┃┣ ┃┃ ┃ ┃┃┗┓
# █ █ █ █ █ █  ┻┛┗┛ ┻ ┗┛┗┛┗┛┻┗┛┛┗  ┻┛┗┛  ┻┛┗┛┗┛┗┛┛ ┗┗┛┛┗ ┻ ┗┛┗┛

#    Entrena un modelo de aprendizaje profundo, y luego guarda el modelo, el tokenizador y el codificador de etiquetas.
#    Pide al usuario un nombre de archivo para guardar el modelo y los objetos asociados.
def entrenar_y_guardar_modelo():
    modelo, tokenizer, label_encoder = entrenar_modelo()    # Entrenar el modelo como antes
    nombre_modelo = simpledialog.askstring("Guardar modelo", "Ingrese el nombre del archivo para guardar el modelo:")  # Pedir al usuario el nombre del archivo para guardar el modelo
    save_model(modelo, f"{nombre_modelo}_modelo.h5")        # Guardar el modelo en un archivo .h5
    with open(f"{nombre_modelo}_tokenizer.pickle", 'wb') as handle:             # Abrir un archivo para guardar el tokenizador
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)        # Guardar el tokenizador en formato pickle
    with open(f"{nombre_modelo}_label_encoder.pickle", 'wb') as handle:         # Abrir un archivo para guardar el codificador de etiquetas
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)    # Guardar el codificador de etiquetas en formato pickle
    return modelo, tokenizer, label_encoder                 # Devolver el modelo, el tokenizador y el codificador de etiquetas

#    Permite cargar un modelo de aprendizaje profundo previamente guardado junto con su tokenizador y codificador de etiquetas.
#    Abre un cuadro de diálogo para seleccionar el archivo del modelo y carga los objetos asociados.
def cargar_modelo():
    filepath = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])           # Abrir un cuadro de diálogo para seleccionar el archivo del modelo
    if filepath:                            # Si se selecciona un archivo
        modelo = load_model(filepath)       # Cargar el modelo desde el archivo
        tokenizer_path = filepath.replace("_modelo.h5", "_tokenizer.pickle")          # Obtener la ruta del archivo del tokenizador
        label_encoder_path = filepath.replace("_modelo.h5", "_label_encoder.pickle")  # Obtener la ruta del archivo del codificador de etiquetas
        with open(tokenizer_path, 'rb') as handle:      # Abrir el archivo del tokenizador
            tokenizer = pickle.load(handle)             # Cargar el tokenizador desde el archivo
        with open(label_encoder_path, 'rb') as handle:  # Abrir el archivo del codificador de etiquetas
            label_encoder = pickle.load(handle)         # Cargar el codificador de etiquetas desde el archivo
        return modelo, tokenizer, label_encoder         # Devolver el modelo, el tokenizador y el codificador de etiquetas
    else:  # Si no se selecciona ningún archivo
        return None, None, None                         # Devolver None para todos los objetos

#    Entrena un modelo de aprendizaje profundo para clasificar discursos en dos categorías: 
#    humanismo y neoliberalismo. La función carga los discursos, preprocesa el texto, 
#    tokeniza y vectoriza los textos, divide los datos en conjuntos de entrenamiento y validación,
#    y entrena un modelo LSTM con los datos procesados.
def entrenar_modelo():  
    # Cargar discursos de humanismo
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")  
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo") 
    # Extraer textos de los discursos
    textos_humanismo = [texto[1] for texto in discursos_humanismo] 
    textos_neoliberalismo = [texto[1] for texto in discursos_neoliberalismo]  

    textos = textos_humanismo + textos_neoliberalismo  # Combinar textos de ambas categorías
    etiquetas = ['humanismo'] * len(textos_humanismo) + ['neoliberalismo'] * len(textos_neoliberalismo)  # Crear etiquetas para los textos

    textos_preprocesados = [' '.join(preprocesar_texto(texto)) for texto in textos]  # Preprocesar los textos

    max_words = 10000               # Definir el número máximo de palabras para el tokenizador
    max_sequence_length = 100       # Definir la longitud máxima de las secuencias

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")       # Crear el tokenizador
    tokenizer.fit_on_texts(textos_preprocesados)                        # Ajustar el tokenizador a los textos preprocesados
    secuencias = tokenizer.texts_to_sequences(textos_preprocesados)     # Convertir los textos a secuencias de enteros
    secuencias_padded = pad_sequences(secuencias, maxlen=max_sequence_length, padding='post', truncating='post')  # Rellenar las secuencias para que tengan la misma longitud

    label_encoder = LabelEncoder()                                      # Crear el codificador de etiquetas
    etiquetas_encoded = label_encoder.fit_transform(etiquetas)          # Codificar las etiquetas en formato numérico
    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(secuencias_padded, etiquetas_encoded, test_size=0.2, random_state=42)  

    modelo = Sequential()                           # Crear un modelo secuencial
    modelo.add(Embedding(max_words, 128, input_length=max_sequence_length))  # Añadir capa de embeddings
    modelo.add(SpatialDropout1D(0.2))               # Añadir capa de dropout espacial
    modelo.add(LSTM(50))                           # Añadir capa LSTM
    #modelo.add(Dropout(0.5))                        # Añadir capa de Dropout para reducir el sobreentrenamiento
    modelo.add(Dense(1, activation='sigmoid'))      # Añadir capa densa con activación sigmoide

    modelo.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])   # Compilar el modelo con pérdida binaria y métrica de precisión

    modelo.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_val, y_val), verbose=2)        # Entrenar el modelo con los datos de entrenamiento

    return modelo, tokenizer, label_encoder         # Devolver el modelo entrenado, el tokenizador y el codificador de etiquetas

#    Clasifica un nuevo documento utilizando un modelo de aprendizaje profundo entrenado.
#    Carga el texto del documento, lo preprocesa, convierte el texto a secuencia, predice su categoría
#    y muestra el resultado en un cuadro de mensaje.
def clasificar_nuevo_documento(filepath, modelo, tokenizer, label_encoder, max_sequence_length):
    with open(filepath, 'r', encoding='utf-8') as file:
        nuevo_texto = file.read()

    # Preprocesar el texto del nuevo documento.
    nuevo_texto_preprocesado = ' '.join(preprocesar_texto(nuevo_texto))
    # Convertir el texto preprocesado a secuencia de enteros.
    nueva_secuencia = tokenizer.texts_to_sequences([nuevo_texto_preprocesado])
    # Rellenar la secuencia para que tenga la longitud máxima.
    nueva_secuencia_padded = pad_sequences(nueva_secuencia, maxlen=max_sequence_length, padding='post', truncating='post')
    # Predecir la categoría del documento.
    prediccion = modelo.predict(nueva_secuencia_padded)
    # Convertir la predicción a una categoría utilizando el codificador de etiquetas.
    categoria = label_encoder.inverse_transform([int(round(prediccion[0][0]))])[0]
    # Mostrar la categoría predicha en un cuadro de mensaje.
    messagebox.showinfo("Clasificación del Documento", f"El documento '{os.path.basename(filepath)}' es clasificado como: {categoria}")


# █ █ █ █ █ █  ┏┓┳┓┏┳┓┳┓┏┓┳┓┏┓  ┳┓┏┓  ┏┳┓┏┓┏┓┓ ┏┓┳┓┏┓
# █ █ █ █ █ █  ┣ ┃┃ ┃ ┣┫┣┫┃┃┣┫  ┃┃┣    ┃ ┣ ┃ ┃ ┣┫┃┃┃┃
# █ █ █ █ █ █  ┗┛┛┗ ┻ ┛┗┛┗┻┛┛┗  ┻┛┗┛   ┻ ┗┛┗┛┗┛┛┗┻┛┗┛

#    Abre una ventana de diálogo para pedir al usuario que ingrese una lista de palabras.
#    Las palabras deben estar separadas por comas y tener más de 3 caracteres. La función devuelve
#    la lista de palabras ingresadas por el usuario.
def pedir_palabras():
    # Ventana para pedir palabras al usuario
    root = tk.Tk()              # Crear una ventana principal de tkinter
    root.withdraw()             # Ocultar la ventana principal
    # Pedir al usuario que ingrese palabras separadas por comas
    palabras_str = simpledialog.askstring("Palabras", "Ingrese las palabras separadas por comas y que estas sean mayores a 3 caracteres:")  
    # Dividir la cadena ingresada en palabras individuales y eliminar espacios adicionales
    palabras = [palabra.strip() for palabra in palabras_str.split(',')]  
    return palabras             # Devolver la lista de palabras ingresadas


# █ █ █ █ █ █  ┏┓┳┳┳┓┏┓┳┏┓┳┓┏┓┏┓  ┏┓┳┓┳┳┓┏┓┳┏┓┏┓┓ ┏┓┏┓
# █ █ █ █ █ █  ┣ ┃┃┃┃┃ ┃┃┃┃┃┣ ┗┓  ┃┃┣┫┃┃┃┃ ┃┃┃┣┫┃ ┣ ┗┓
# █ █ █ █ █ █  ┻ ┗┛┛┗┗┛┻┗┛┛┗┗┛┗┛  ┣┛┛┗┻┛┗┗┛┻┣┛┛┗┗┛┗┛┗┛

#    Muestra características generales de los discursos de humanismo o neoliberalismo.
#    Carga los discursos, los preprocesa, calcula estadísticas y las muestra en una ventana de tkinter.
def mostrar_caracteristicas_generales(tipo):
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")                        # Cargar discursos de humanismo
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")              # Cargar discursos de neoliberalismo
    textos_preprocesados_humanismo = [preprocesar_texto(texto[1]) for texto in discursos_humanismo]             # Preprocesar textos de humanismo
    textos_preprocesados_neoliberalismo = [preprocesar_texto(texto[1]) for texto in discursos_neoliberalismo]   # Preprocesar textos de neoliberalismo
    estadisticas_humanismo = calcular_estadisticas(textos_preprocesados_humanismo)                              # Calcular estadísticas de textos de humanismo
    estadisticas_neoliberalismo = calcular_estadisticas(textos_preprocesados_neoliberalismo)                    # Calcular estadísticas de textos de neoliberalismo

    def mostrar_en_ventana(titulo, estadisticas):
        ventana = tk.Toplevel(root)     # Crear una ventana secundaria
        ventana.title(titulo)           # Establecer el título de la ventana

        # Preparar el texto con las estadísticas
        texto = f"Características Generales - {titulo}\n\n"
        texto += f"Total de palabras: {estadisticas['total_palabras']}\n"
        texto += f"Promedio de palabras por documento: {estadisticas['promedio_palabras']}\n"
        texto += f"Longitud del documento más largo: {estadisticas['longitud_maxima']}\n"
        texto += f"Longitud del documento más corto: {estadisticas['longitud_minima']}\n"
        texto += f"Palabras más frecuentes: {estadisticas['palabras_frecuentes']}\n"
        texto += f"Longitudes: {[len(texto) for texto in textos_preprocesados_humanismo] if tipo == 1 else [len(texto) for texto in textos_preprocesados_neoliberalismo]}\n"
        texto += f"Palabras únicas: {len(set([palabra for texto in textos_preprocesados_humanismo for palabra in texto])) if tipo == 1 else len(set([palabra for texto in textos_preprocesados_neoliberalismo for palabra in texto]))}\n"

        label = tk.Label(ventana, text=texto, justify="left", padx=10, pady=10)     # Crear una etiqueta con el texto
        label.pack()                                                                # Empaquetar la etiqueta en la ventana

    if tipo == 1:   # Si el tipo es 1, mostrar estadísticas de humanismo
        mostrar_en_ventana("Humanismo", estadisticas_humanismo)
    else:           # Si el tipo no es 1, mostrar estadísticas de neoliberalismo
        mostrar_en_ventana("Neoliberalismo", estadisticas_neoliberalismo)


#    Muestra un gráfico de dispersión léxica para discursos de humanismo o neoliberalismo.
#    Carga los discursos, los preprocesa, pide palabras al usuario y genera el gráfico de dispersión léxica.
def mostrar_grafico_dispersion_lexica(tipo):
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")                        # Cargar discursos de humanismo
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")              # Cargar discursos de neoliberalismo
    textos_preprocesados_humanismo = [preprocesar_texto(texto[1]) for texto in discursos_humanismo]             # Preprocesar textos de humanismo
    textos_preprocesados_neoliberalismo = [preprocesar_texto(texto[1]) for texto in discursos_neoliberalismo]   # Preprocesar textos de neoliberalismo
    palabras_a_graficar = pedir_palabras()  # Pedir palabras al usuario para graficar
    if tipo == 1:   # Si el tipo es 1, graficar dispersión léxica para humanismo
        graficar_dispersion_lexica(textos_preprocesados_humanismo, palabras_a_graficar, 1)
    else:           # Si el tipo no es 1, graficar dispersión léxica para neoliberalismo
        graficar_dispersion_lexica(textos_preprocesados_neoliberalismo, palabras_a_graficar, 0)


#    Muestra un gráfico de series de tiempo para discursos de humanismo o neoliberalismo.
#    Carga los discursos, pide palabras al usuario y genera el gráfico de series de tiempo.
def mostrar_grafico_series_tiempo(tipo):
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")            # Cargar discursos de humanismo
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")  # Cargar discursos de neoliberalismo
    palabras_a_graficar = pedir_palabras()  # Pedir palabras al usuario para graficar
    if tipo == 1:       # Si el tipo es 1, graficar series de tiempo para humanismo
        graficar_series_tiempo(discursos_humanismo, palabras_a_graficar, 1)
    else:               # Si el tipo no es 1, graficar series de tiempo para neoliberalismo
        graficar_series_tiempo(discursos_neoliberalismo, palabras_a_graficar, 0)


#    Muestra una nube de palabras (WordCloud) para discursos de humanismo o neoliberalismo.
#    Carga los discursos, los preprocesa y genera la nube de palabras correspondiente.
def mostrar_wordcloud(tipo):
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")                        # Cargar discursos de humanismo
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")              # Cargar discursos de neoliberalismo
    textos_preprocesados_humanismo = [preprocesar_texto(texto[1]) for texto in discursos_humanismo]             # Preprocesar textos de humanismo
    textos_preprocesados_neoliberalismo = [preprocesar_texto(texto[1]) for texto in discursos_neoliberalismo]   # Preprocesar textos de neoliberalismo
    if tipo == 1:   # Si el tipo es 1, generar nube de palabras para humanismo
        generar_wordcloud(textos_preprocesados_humanismo, 1)
    else:           # Si el tipo no es 1, generar nube de palabras para neoliberalismo
        generar_wordcloud(textos_preprocesados_neoliberalismo, 0)


#    Muestra un gráfico de frecuencia de palabras para discursos de humanismo o neoliberalismo.
#    Carga los discursos, los preprocesa, pide palabras al usuario y genera el gráfico de frecuencia de palabras.
def mostrar_grafico_frecuencia_palabras(tipo):
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")                        # Cargar discursos de humanismo
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")              # Cargar discursos de neoliberalismo
    textos_preprocesados_humanismo = [preprocesar_texto(texto[1]) for texto in discursos_humanismo]             # Preprocesar textos de humanismo
    textos_preprocesados_neoliberalismo = [preprocesar_texto(texto[1]) for texto in discursos_neoliberalismo]   # Preprocesar textos de neoliberalismo
    palabras_a_graficar = pedir_palabras()  # Pedir palabras al usuario para graficar
    if tipo == 1:   # Si el tipo es 1, graficar frecuencia de palabras para humanismo
        graficar_frecuencia_palabras(textos_preprocesados_humanismo, palabras_a_graficar, 1)
    else:           # Si el tipo no es 1, graficar frecuencia de palabras para neoliberalismo
        graficar_frecuencia_palabras(textos_preprocesados_neoliberalismo, palabras_a_graficar, 0)


#    Extrae y guarda el vocabulario de discursos de humanismo y neoliberalismo.
#    Carga los discursos, los preprocesa, extrae el vocabulario y lo guarda en archivos de texto.
def vocabulario():
    discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")                        # Cargar discursos de humanismo
    discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")              # Cargar discursos de neoliberalismo
    textos_preprocesados_humanismo = [preprocesar_texto(texto[1]) for texto in discursos_humanismo]             # Preprocesar textos de humanismo
    textos_preprocesados_neoliberalismo = [preprocesar_texto(texto[1]) for texto in discursos_neoliberalismo]   # Preprocesar textos de neoliberalismo
    vocabulario_humanismo = extraer_vocabulario(textos_preprocesados_humanismo)                                 # Extraer vocabulario de humanismo
    vocabulario_neoliberalismo = extraer_vocabulario(textos_preprocesados_neoliberalismo)                       # Extraer vocabulario de neoliberalismo
    guardar_vocabulario(vocabulario_humanismo, 'vocabulario_humanismo.txt')                 # Guardar vocabulario de humanismo en un archivo de texto
    guardar_vocabulario(vocabulario_neoliberalismo, 'vocabulario_neoliberalismo.txt')       # Guardar vocabulario de neoliberalismo en un archivo de texto
    vocabulario_corpus = vocabulario_neoliberalismo.union(vocabulario_humanismo)            # Combinar vocabularios usando union()
    guardar_vocabulario(vocabulario_corpus, 'corpus.txt')                                   # Guardar corpus en un archivo de texto


#    Permite al usuario seleccionar un archivo de texto y clasificarlo utilizando un modelo previamente entrenado.
#    Abre un cuadro de diálogo para seleccionar el archivo y luego clasifica el documento seleccionado.
def cargar_y_clasificar_documento():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])                       # Abrir un cuadro de diálogo para seleccionar un archivo de texto
    if filepath:  # Si se selecciona un archivo
        clasificar_nuevo_documento(filepath, modelo, tokenizer, label_encoder, max_sequence_length)  # Clasificar el documento seleccionado


# █ █ █ █ █ █  ┏┓┳┳┳┓┏┓┳┏┓┳┓┏┓┏┓  ┳┳┓┳┏┓┳┏┓┓ ┏┓┏┓
# █ █ █ █ █ █  ┣ ┃┃┃┃┃ ┃┃┃┃┃┣ ┗┓  ┃┃┃┃┃ ┃┣┫┃ ┣ ┗┓
# █ █ █ █ █ █  ┻ ┗┛┛┗┗┛┻┗┛┛┗┗┛┗┛  ┻┛┗┻┗┛┻┛┗┗┛┗┛┗┛

#    Permite al usuario elegir entre cargar un modelo existente o entrenar uno nuevo.
#    Muestra un cuadro de mensaje para la decisión del usuario y actúa en consecuencia.
respuesta = messagebox.askyesno("Modelo", "¿Desea cargar un modelo existente?")  # Preguntar al usuario si desea cargar un modelo existente
if respuesta:               # Si la respuesta es sí
    modelo, tokenizer, label_encoder = cargar_modelo()  # Cargar el modelo existente
    if modelo is None:      # Si no se cargó un modelo
        messagebox.showinfo("Error", "No se seleccionó un modelo. Se entrenará uno nuevo.")     # Mostrar mensaje de error
        modelo, tokenizer, label_encoder = entrenar_y_guardar_modelo()                          # Entrenar y guardar un nuevo modelo
else:                       # Si la respuesta es no
    modelo, tokenizer, label_encoder = entrenar_y_guardar_modelo()                              # Entrenar y guardar un nuevo modelo

max_sequence_length = 100   # Definir la longitud máxima de las secuencias

discursos_humanismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\humanismo")            # Cargar discursos de humanismo
discursos_neoliberalismo = cargar_discursos(r"A:\Principal\Escritorio\xd\fold\neoliberalismo")  # Cargar discursos de neoliberalismo

vocabulario()  # Extraer y guardar el vocabulario de los discursos cargados


# █ █ █ █ █ █  ┳┳┓┏┳┓┏┓┳┓┏┓┏┓┏┓  ┏┓┳┓┏┓┏┓┳┏┓┏┓
# █ █ █ █ █ █  ┃┃┃ ┃ ┣ ┣┫┣ ┣┫┏┛  ┃┓┣┫┣┫┣ ┃┃ ┣┫
# █ █ █ █ █ █  ┻┛┗ ┻ ┗┛┛┗┻ ┛┗┗┛  ┗┛┛┗┛┗┻ ┻┗┛┛┗

#    Crea la interfaz gráfica de usuario para el análisis de discursos de humanismo.
#    Inicializa la ventana principal, configura el marco y los botones correspondientes a las distintas
#    funcionalidades de análisis.
root = tk.Tk()                                   # Crear la ventana principal de tkinter
root.title("Sistema de Análisis Político")       # Establecer el título de la ventana

frame_humanismo = ttk.Frame(root, padding="10")  # Crear un marco con padding de 10 unidades
frame_humanismo.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))  # Colocar el marco en la ventana principal

titulo_humanismo = ttk.Label(frame_humanismo, text="Humanismo", font=("Helvetica", 16))     # Crear una etiqueta con el título "Humanismo"
titulo_humanismo.grid(row=0, column=0, columnspan=2, pady=10)                               # Colocar la etiqueta en el marco

# Crear botones para las distintas funcionalidades de análisis y colocarlos en el marco
ttk.Button(frame_humanismo, text="Ver Características Generales",          command=lambda: mostrar_caracteristicas_generales(1)  ).grid(row=1, column=0, pady=5)
ttk.Button(frame_humanismo, text="Ver Gráfico de Dispersión Léxica",       command=lambda: mostrar_grafico_dispersion_lexica(1)  ).grid(row=2, column=0, pady=5)
ttk.Button(frame_humanismo, text="Ver Gráficos de Series de Tiempo",       command=lambda: mostrar_grafico_series_tiempo(1)      ).grid(row=3, column=0, pady=5)
ttk.Button(frame_humanismo, text="Ver Wordclouds",                         command=lambda: mostrar_wordcloud(1)                  ).grid(row=4, column=0, pady=5)
ttk.Button(frame_humanismo, text="Ver Gráficos de Frecuencia de Palabras", command=lambda: mostrar_grafico_frecuencia_palabras(1)).grid(row=5, column=0, pady=5)

# Crea la interfaz gráfica de usuario para el análisis de discursos de neoliberalismo.
# Inicializa un marco y los botones correspondientes a las distintas funcionalidades de análisis.
# Además, añade un botón para cargar y clasificar un nuevo documento, y ejecuta el bucle principal de la interfaz.
frame_neoliberalismo = ttk.Frame(root, padding="10")                         # Crear un marco con padding de 10 unidades
frame_neoliberalismo.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))  # Colocar el marco en la ventana principal

titulo_neoliberalismo = ttk.Label(frame_neoliberalismo, text="Neoliberalismo", font=("Helvetica", 16))  # Crear una etiqueta con el título "Neoliberalismo"
titulo_neoliberalismo.grid(row=0, column=0, columnspan=2, pady=10)                                      # Colocar la etiqueta en el marco

# Crear botones para las distintas funcionalidades de análisis y colocarlos en el marco
ttk.Button(frame_neoliberalismo, text="Ver Características Generales",          command=lambda: mostrar_caracteristicas_generales(0)  ).grid(row=1, column=0, pady=5)
ttk.Button(frame_neoliberalismo, text="Ver Gráfico de Dispersión Léxica",       command=lambda: mostrar_grafico_dispersion_lexica(0)  ).grid(row=2, column=0, pady=5)
ttk.Button(frame_neoliberalismo, text="Ver Gráficos de Series de Tiempo",       command=lambda: mostrar_grafico_series_tiempo(0)      ).grid(row=3, column=0, pady=5)
ttk.Button(frame_neoliberalismo, text="Ver Wordclouds",                         command=lambda: mostrar_wordcloud(0)                  ).grid(row=4, column=0, pady=5)
ttk.Button(frame_neoliberalismo, text="Ver Gráficos de Frecuencia de Palabras", command=lambda: mostrar_grafico_frecuencia_palabras(0)).grid(row=5, column=0, pady=5)

# Crear un botón para cargar y clasificar un nuevo documento y colocarlo en la ventana principal
ttk.Button(root, text="Cargar y Clasificar Nuevo Documento", command=cargar_y_clasificar_documento).grid(row=1, column=0, columnspan=2, pady=20)

root.mainloop()  # Ejecutar el bucle principal de la interfaz
