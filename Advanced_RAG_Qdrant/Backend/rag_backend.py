from .chunking import recursive_char_split_chunking
from dotenv import load_dotenv
import os
load_dotenv()

CHUNK_SIZE = int(os.environ['CHUNK_SIZE'])
CHUNK_OVERLAP = int(os.environ['CHUNK_OVERLAP'])
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']

def chunk_uploaded_docs(documents):
    chunks = recursive_char_split_chunking(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    document_source = documents[0].metadata.get('source', 'Unknown') if documents else 'Unknown'
    print(f"Document source : {document_source}, no.of chunks created : {len(chunks)}.")
    return chunks

def create_embeddings():
    ...