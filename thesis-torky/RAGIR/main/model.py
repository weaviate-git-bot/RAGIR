import os

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import LlamaCPP
from llama_index.callbacks import CallbackManager, LlamaDebugHandler


def create_reader(directory: str) -> SimpleDirectoryReader:
    """Create a SimpleDirectoryReader object to read documents from a given directory.

    Args:
        directory (str): The directory path where the documents are located.

    Returns:
        SimpleDirectoryReader: A SimpleDirectoryReader object for the given directory.
    """

    reader = SimpleDirectoryReader(
        input_dir=directory, required_exts=[".pdf"], filename_as_id=True
    )

    docs = reader.load_data()

    return docs


def create_nodes(
    docs: SimpleDirectoryReader,
    model_path: str,
    local_llm: bool = True,
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
) -> VectorStoreIndex:
    """Create a VectorStoreIndex based on the provided SimpleDirectoryReader object.

    Args:
        docs (SimpleDirectoryReader): The SimpleDirectoryReader object containing the documents.
        local_llm (bool, optional): A flag indicating whether to use local llama. Defaults to True.
        chunk_size (int, optional): Size of the chunks. Defaults to 1024.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 20.

    Returns:
        VectorStoreIndex: A VectorStoreIndex object created based on the provided parameters.
    """

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    if local_llm:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0.5,
            max_new_tokens=1024,
            context_window=3900,
            model_kwargs={"n_gpu_layers": 2},
            verbose=True,
        )
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
            callback_manager=callback_manager,
        )
    index = VectorStoreIndex(docs, show_progress=True, service_context=service_context)
    return index, service_context


def create_index(doc_directory: str, index_path: str, model_path: str) -> None:
    """
    Creates the index for documents located in the specified directory.

    Args:
        doc_directory (str): The directory path where the documents are located.
        index_path (str): The path to the index storage directory.
        model_path (str): The path to the LLM model

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified directory or file does not exist.

    """
    dir = os.listdir(index_path)
    if not len(dir):
        docs = create_reader(directory=doc_directory)
        index, service_context = create_nodes(
            docs, chunk_size=1024, chunk_overlap=20, model_path=model_path
        )
        index.storage_context.persist(persist_dir=index_path)

    else:
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0,
            max_new_tokens=1024,
            context_window=3900,
            model_kwargs={"n_gpu_layers": 2},
            verbose=True,
        )
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        service_context = ServiceContext.from_defaults(
            llm=llm,
            chunk_size=1024,
            embed_model="local",
            callback_manager=callback_manager,
        )
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(
            storage_context, service_context=service_context
        )
        print("Index Already Exists")

    return index, service_context
