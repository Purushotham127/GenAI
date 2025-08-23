import pymongo
import os
import dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
one_dir_up = os.path.abspath(os.path.join(current_dir, "../"))
env_path = os.path.join(one_dir_up, '.env')

dotenv.load_dotenv(env_path)

database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")
def init_connection():
    return pymongo.MongoClient(host = "localhost",port = 27017)

client = init_connection()