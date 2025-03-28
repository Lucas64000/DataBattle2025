import pickle

from qdrant_client import QdrantClient

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, QueryBundle

class Qdrant_DB:
    def __init__(self, client: QdrantClient, collection_name="data-battle-v3") -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_store = self.init_qdrant_vector_store()
        self.vector_index = self.init_vector_index()
        self.retriever = self.init_retriever()
        
    # Init Qdrant vector store
    def init_qdrant_vector_store(self):
        return QdrantVectorStore(client=self.client, collection_name=self.collection_name)

    # create an index
    def init_vector_index(self):
        return VectorStoreIndex.from_vector_store(self.vector_store)
        
    def init_retriever(self, similarity_top_k=2):
        return self.vector_index.as_retriever(similarity_top_k=similarity_top_k)
    
    def add_nodes(self, nodes):
        self.vector_store.upsert(nodes)
        self.vector_index = self.init_vector_index()
        self.retriever = self.init_retriever()
        
    def get_nodes(self):
        return self.vector_store.get_nodes()

    def _retrieve(self, query_bundle: QueryBundle):
        return self.retriever.retrieve(query_bundle)
