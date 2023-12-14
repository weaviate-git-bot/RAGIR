import weaviate
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.prompts import PromptTemplate
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.node_parser import SentenceSplitter
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from RAGIR.utils.util_functions import (
    initialize_service_context,
    initialize_embedding
)


class WeaviateVectorDB:
    def __init__(self, client_url, embedding_model, index_name, documents_path) -> None:
        self.client_url = client_url
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.documents_path = documents_path
        self.client = weaviate.Client(self.client_url)
        self.nodes = self.create_chunked_nodes(chunk_size=200, chunk_overlap=20)
        self.index = self.read_index(text_key="content")

    def create_chunked_nodes(self, chunk_size: int, chunk_overlap: int) -> list:
        documents = SimpleDirectoryReader(self.documents_path).load_data()
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.get_nodes_from_documents(documents)

    def create_storage_context(self, vector_store) -> StorageContext:
        return StorageContext.from_defaults(vector_store=vector_store)

    def create_service_context(self) -> ServiceContext:
        return initialize_service_context(
            embed_model=initialize_embedding(model_name=self.embedding_model)
        )

    def create_index(self, chunk_size: int, chunk_overlap: int, text_key: str):
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client, index_name=self.index_name, text_key=text_key
        )
        nodes = self.create_chunked_nodes(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        storage_context = self.create_storage_context(vector_store=vector_store)
        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            service_context=self.create_service_context(),
        )

    def read_index(self, text_key: str):
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client, index_name=self.index_name, text_key=text_key
        )
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=self.create_service_context(),
        )

    def get_index(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        try:
            self.client.schema.get(self.index_name)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            return self.create_index(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, text_key="content"
            )
        else:
            return self.read_index(text_key="content")

    def get_query_engine(
        self,
        prompt_template: PromptTemplate,
        service_context: ServiceContext,
        similarity_top_k: int = 3,
        top_n: int = 3,
        alpha: int = 0.75,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        include_postprocessor: bool = True,
    ):
        index = self.get_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        query_engine = index.as_query_engine(
            prompt_template=prompt_template,
            similarity_top_k=similarity_top_k,
            alpha=alpha,
            service_context=service_context,
        )

        if include_postprocessor:
            query_engine.postprocessor = FlagEmbeddingReranker(
                top_n=top_n, model="BAAI/bge-reranker-large"
            )

        return query_engine
