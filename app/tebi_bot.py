# Importar las bibliotecas necesarias
import json
import pickle
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from unidecode import unidecode
from app.configdb import get_mongo_client

# Descargar los recursos de NLTK si no están disponibles
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo del chatbot previamente entrenado

model = load_model('archive/chatbot_model.h5')

# Inicializar el lematizador de palabras
lemmatizer = WordNetLemmatizer()

# Cargar las palabras y clases previamente procesadas
words = pickle.load(open('archive/words.pkl', 'rb'))
classes = pickle.load(open('archive/classes.pkl', 'rb'))

# Función para limpiar la frase de entrada y convertirla en una bolsa de palabras
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenizar la frase en palabras
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lematizar y normalizar palabras
    return sentence_words

# Función para crear una bolsa de palabras (bag of words) a partir de la frase de entrada
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)  # Obtener palabras lematizadas y normalizadas
    bag = [0]*len(words)  # Inicializar la bolsa de palabras con ceros
    for s in sentence_words:
        s = unidecode(s.lower())
        if s in words:
            bag[words.index(s)] = 1  # Establecer 1 en la posición correspondiente si la palabra está en words
            if show_details:
                print(f"found in bag: {s}")
    return np.array(bag)  # Devolver la bolsa de palabras como un array numpy

# Función para predecir la clase de la frase de entrada y obtener la respuesta del chatbot
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)  # Obtener la bolsa de palabras de la frase de entrada
    res = model.predict(np.array([p]))[0]  # Predecir la clase utilizando el modelo
    ERROR_THRESHOLD = 0.65  # Umbral de error para aceptar la predicción
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filtrar resultados por encima del umbral
    
    results.sort(key=lambda x: x[1], reverse=True)  # Ordenar resultados por probabilidad descendente
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})  # Agregar la intención y probabilidad a return_list
    return return_list  # Devolver la lista de intenciones con sus probabilidades

# Función para obtener la respuesta del chatbot basada en la intención predicha
def get_response(ints):
    # Obtener la etiqueta de la intención predicha
    tag = ints[0]['intent'] if len(ints) > 0 else 'error'
    
    responses = get_responses_from_DB(tag)
    result = random.choice(responses['responses'])  # Elegir una respuesta aleatoria de las respuestas disponibles
    return result  # Devolver la respuesta seleccionada

def get_responses_from_DB(tag):
    collection_name = 'interactions'
    mongo_client, db = get_mongo_client()
    collection = mongo_client['interactions']
    
    collection = db[collection_name]
    return collection.find_one({'tag': tag})

# Función principal para obtener la respuesta del chatbot dada una entrada de usuario
def chatbot_response(text):
    ints = predict_class(text, model)  # Predecir la intención de la frase de entrada
    res = get_response(ints)  # Obtener la respuesta del chatbot basada en la intención predicha
    return res  # Devolver la respuesta del chatbot

def response_to_user(message):
   return chatbot_response(message)
