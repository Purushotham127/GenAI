from .Qdrant_util import QdrantVectorDB
from dotenv import load_dotenv
import os
load_dotenv()

qdrant_host = os.environ['QDRANT_HOST']
qdrant_port = os.environ['QDRANT_PORT']

qdrant_db = QdrantVectorDB(qdrant_host, qdrant_port)