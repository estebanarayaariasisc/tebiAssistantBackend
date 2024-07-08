import pickle, numpy, random, nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from unidecode import unidecode
from flask import current_app as app
from app.configdb import get_mongo_client


def start_up_training_nodel():

    # Descargar recursos necesarios de NLTK si no están disponibles
    nltk.download('punkt')
    nltk.download('wordnet')

    # Inicializar lematizador de palabras
    lemmatizer = WordNetLemmatizer()

    # Cargar los parametros de preguntas desde Mongo
    collection_name = 'interactions'
    mongo_client, db = get_mongo_client()
    collection = mongo_client['interactions']
    
    collection = db[collection_name]
    intents = collection.find()

    # Preprocesamiento de datos
    words, classes, documents = []
    ignore_letters = ['!', '?', ',', '.', '¡', '¿', '*', '-', '_', ' ']

    for intent in intents:
        for pattern in intent['patterns']:
            # Tokenizar las palabras en el patrón
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list) 
            documents.append((word_list, intent['tag']))
            
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lematizar y normalizar las palabras
    words = [lemmatizer.lemmatize(unidecode(word.lower())) for word in words if word not in ignore_letters]
    words = sorted(set(words))  
    classes = sorted(set(classes)) 

    # Guardar las palabras y categorias
    pickle.dump(words, open('archive/words.pkl', 'wb'))
    pickle.dump(classes, open('archive/classes.pkl', 'wb'))

    # Preparar los datos de entrenamiento
    training = [] 
    output_empty = [0] * len(classes) 

    for doc in documents:
        bag = []  
        pattern_words = doc[0]  
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words] 
        
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)  
        
        output_row = list(output_empty)  
        output_row[classes.index(doc[1])] = 1  
        
        training.append([bag, output_row])

    random.shuffle(training)

    # Separar los datos de entrada (X) y las salidas esperadas (y)
    train_x = numpy.array([i[0] for i in training])  
    train_y = numpy.array([i[1] for i in training]) 
    print("Forma de train_x:", train_x.shape)
    print("Forma de train_y:", train_y.shape)

    # configurar y compilar el modelo de red neuronal
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(64, activation='relu')) 
    model.add(Dropout(0.5))  
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Se guarda el modelo
    model.save('archive/chatbot_model.h5', hist)
    print("Modelo creado exitosamente")
