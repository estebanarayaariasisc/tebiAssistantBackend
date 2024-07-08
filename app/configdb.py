import urllib.parse
from pymongo import MongoClient

username = urllib.parse.quote_plus('estebanarayaariasisc')
password = urllib.parse.quote_plus('2590Nabetse:01')
MONGO_URI = f"mongodb+srv://{username}:{password}@tebicluster.kx6ykad.mongodb.net/?retryWrites=true&w=majority&appName=TebiCluster"
MONGO_DB_NAME = 'interactions'

# Función para establecer y obtener la conexión a MongoDB
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        print("MongoDB connection successful.")
        return client, db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None, None
