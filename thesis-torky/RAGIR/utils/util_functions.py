import logging
import sys
from typing import List, Optional

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.evaluation import (
    DatasetGenerator,
)
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
)


def initialize_llama_cpp(
    model_path: str,
    temperature: float,
    max_new_tokens: int,
    context_window: int,
    model_kwargs: dict,
    verbose: bool,
) -> LlamaCPP:
    """
    Initialize the LlamaCPP object.

    Args:
    - model_path: Path to the LlamaCPP model.
    - temperature: Temperature for the model.
    - max_new_tokens: Maximum number of new tokens.
    - context_window: Context window size.
    - model_kwargs: Additional keyword arguments for the model.
    - verbose: Verbosity flag.

    Returns:
    - Initialized LlamaCPP object.
    """
    return LlamaCPP(
        model_path=model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        model_kwargs=model_kwargs,
        verbose=verbose,
    )


def load_documents(data_path: str) -> List[str]:
    """
    Load documents from a directory.

    Args:
    - data_path: Path to the directory containing documents.

    Returns:
    - List of documents loaded from the directory.
    """
    reader = SimpleDirectoryReader(data_path)
    return reader.load_data()


def initialize_embedding(model_name: str) -> HuggingFaceEmbedding:
    """
    Initialize the HuggingFace embedding model.

    Args:
    - model_name: Name of the HuggingFace model.

    Returns:
    - Initialized HuggingFaceEmbedding object.
    """
    return HuggingFaceEmbedding(model_name=model_name)


def initialize_service_context(
    llm: Optional[LlamaCPP] = None, embed_model: Optional[HuggingFaceEmbedding] = None
) -> ServiceContext:
    """
    Initialize the ServiceContext object.

    Args:
    - llm: Optional LlamaCPP object.
    - embed_model: Optional HuggingFaceEmbedding object.

    Returns:
    - Initialized ServiceContext object.
    """
    if llm is not None and embed_model is not None:
        return ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    else:
        return ServiceContext.from_defaults(llm=None, embed_model=embed_model)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def generate_dataset(
    documents: List[str], service_context: ServiceContext, num_questions_per_chunk: int
) -> DatasetGenerator:
    """
    Generate a dataset for evaluation.

    Args:
    - documents: List of documents.
    - service_context: ServiceContext object.
    - num_questions_per_chunk: Number of questions per chunk.

    Returns:
    - Initialized DatasetGenerator object.
    """
    return DatasetGenerator.from_documents(
        documents,
        service_context=service_context,
        num_questions_per_chunk=num_questions_per_chunk,
    )
