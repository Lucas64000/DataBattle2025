from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, QueryBundle, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List
import Stemmer
import gc

import database

class EmbeddingBM25RerankerRetriever(BaseRetriever) :
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: SentenceTransformerRerank,
    ) -> None:
        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        vector_nodes.extend(bm25_nodes)

        retrieved_nodes = self.reranker.postprocess_nodes(
            vector_nodes, query_bundle
        )

        return retrieved_nodes

# create the Qdrant retriever 
def create_embedding_retriever(qdrant_db: database.Qdrant_DB, similarity_top_k=50) :
    return qdrant_db.init_retriever(similarity_top_k)

# create the bm25 retriever for a given list of nodes
def create_bm25_retriever(nodes, similarity_top_k=15) :
    return BM25Retriever.from_defaults(
        nodes = nodes,
        similarity_top_k = similarity_top_k,
        stemmer = Stemmer.Stemmer("english"),
        language = "english",
    )

# create the full retriever 
def create_full_retriever(qdrant_db: database.Qdrant_DB, emb_k=50, bm25_k=15, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_n=6) :
    # create the embedding retriever 
    embedding_retriever = create_embedding_retriever(qdrant_db, similarity_top_k=emb_k)

    # get nodes for the BM25 retriever
    nodes = qdrant_db.get_nodes()

    # create the BM25 retriever
    bm25_retriever = create_bm25_retriever(
        nodes, similarity_top_k=bm25_k
    )

    # create the reranker
    reranker = SentenceTransformerRerank(
        model=reranker_model, top_n=reranker_n
    )

    # fusion retrievers and add the reranker
    embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
        embedding_retriever, bm25_retriever, reranker=reranker
    )
    
    return embedding_bm25_retriever_rerank

# create the retriever based on the retriever_name argument
def create_chosen_retriever(qdrant_db: database.Qdrant_DB, retriever_name="Embedding/BM25 retrievers + rerank", emb_k=50, bm25_k=15, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_n=6) :
    # create the embedding retriever 
    if retriever_name == "Embedding retriever":
        return create_embedding_retriever(qdrant_db, similarity_top_k=emb_k)

    # create the BM25 retriever
    elif retriever_name == "BM25 retriever":
        nodes = qdrant_db.get_nodes()
        return create_bm25_retriever(nodes, similarity_top_k=bm25_k)

    # create the full retriever
    else:
        return create_full_retriever(qdrant_db, emb_k=emb_k, bm25_k=bm25_k, reranker_model=reranker_model, reranker_n=reranker_n)

# retrieve context and sources 
def retrieve_context_and_sources(retriever, query, show_scores=False):
    retrieved_nodes = retriever.retrieve(query)

    # concatenate the chunks to have the context 
    context = "\n*********\n".join([node.text for node in retrieved_nodes])

    # get sources from the chunks 
    sources = []
    for node in retrieved_nodes:
        file_name = node.metadata.get("file_name", "Inconnu")  
        page_label = node.metadata.get("page_label")
        
        if page_label:
            sources.append(f"{file_name} (Page {page_label})")
        else:
            sources.append(file_name)
        if show_scores:
            sources.append(f" - Score: {node.score:.2f}")

    return context, sources