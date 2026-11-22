from qdrant_client import QdrantClient

class QdrantVectorDB:
    '''Qdrant vector database class'''
    def __init__(self, host, port, grpc_port = 6334):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port

    def create_client(self):
        client = QdrantClient(
            host=self.host, port=self.port
        )
        self.client = client

    def create_db_collection(self, collection_name, vectors_config):
        self.collection_name = collection_name
        self.vectors_config = vectors_config
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vectors_config
        )

    def get_db_collections(self):
        return self.client.get_collections()

    