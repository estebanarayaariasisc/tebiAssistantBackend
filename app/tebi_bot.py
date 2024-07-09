# Importar las bibliotecas necesarias
import json, pickle, numpy, random, nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from unidecode import unidecode
from app.configdb import get_mongo_client

# Descargar los recursos de NLTK si no están disponibles
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo entrenado
model = load_model('archive/chatbot_model.h5')

# Inicializar el lematizador de palabras
lemmatizer = WordNetLemmatizer()

# Cargar las palabras y categorias procesadas
words = pickle.load(open('archive/words.pkl', 'rb'))
classes = pickle.load(open('archive/classes.pkl', 'rb'))

# Funcion que devuelve la respuesta al cliente
def response_to_user(message):
   return chatbot_response(message)

def chatbot_response(text):
    ints = predict_class(text, model) 
    res = get_response(ints) 
    return res  

# Función para predecir la categoría de respuesta
def predict_class(sentence, model):
    p = createBag(sentence, words, show_details=False)  
    res = model.predict(numpy.array([p]))[0] 
    ERROR_THRESHOLD = 0.65  
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] 
    
    results.sort(key=lambda x: x[1], reverse=True)  
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])}) 
    return return_list  

# Función para crear una bolsa de palabras a partir de la frase de entrada
def createBag(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)  
    bag = [0]*len(words)  
    for s in sentence_words:
        s = unidecode(s.lower())
        if s in words:
            bag[words.index(s)] = 1 
            if show_details:
                print(f"found in bag: {s}")
    return numpy.array(bag) 

# Función para limpiar la frase de entrada y convertirla en una bolsa de palabras
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  
    return sentence_words

# Función para obtener la respuesta según la categoría
def get_response(ints):
    # Obtener la etiqueta de la categoría
    tag = ints[0]['intent'] if len(ints) > 0 else 'error'
    
    responses = get_responses_from_DB(tag)
    result = random.choice(responses['responses'])  
    return result

# Funcion que obtiene la respuesta desde Mongo
def get_responses_from_DB(tag):
    collection_name = 'interactions'
    mongo_client, db = get_mongo_client()
    collection = mongo_client['interactions']
    
    collection = db[collection_name]
    return collection.find_one({'tag': tag})

