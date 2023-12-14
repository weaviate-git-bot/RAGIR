from llama_index.prompts import PromptTemplate
from llama_index import ServiceContext
from RAGIR.database.setup_database import WeaviateVectorDB
from RAGIR.utils.util_functions import initialize_embedding, initialize_llama_cpp


def create_text_qa_template() -> PromptTemplate:
    """
    Create a text QA template for querying.
    """
    text_qa_template_str = (
        "Only use the context information given below."
        "\n---------------------\n{context_str}\n---------------------\n"
        " , answer the following query {query_str}.\n"
        "If you didn't find the answer within the context"
        "reply with I don't know. After that reference which paper you've used.\n"
    )
    return PromptTemplate(text_qa_template_str)


def main() -> None:
    """
    Main function to execute the script.
    """

    client_url = "http://localhost:8080"
    embedding_model = "BAAI/bge-small-en-v1.5"
    model_path = "/Users/torky/Documents/LlamaCPP-2/zephyr-7b-beta.Q4_0.gguf"
    documents_path = "/Users/torky/Documents/thesis-torky/docs/research_papers"
    index_name = "IrAnthology"

    weaviate_db = WeaviateVectorDB(
        client_url=client_url,
        embedding_model=embedding_model,
        index_name=index_name,
        documents_path=documents_path,
    )

    text_qa_template = create_text_qa_template()

    query_text = (
        "What are the ways to get rid of Hallucinations in Large Language Models"
    )

    embedding_model = initialize_embedding(model_name=embedding_model)

    llm = initialize_llama_cpp(
        model_path=model_path,
        temperature=0,
        max_new_tokens=1024,
        context_window=3900,
        model_kwargs={"n_gpu_layers": 2},
        verbose=False,
    )

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)

    query_engine = weaviate_db.get_query_engine(
        prompt_template=text_qa_template,
        alpha=1,
        service_context=service_context,
        similarity_top_k=3,
    )

    response = query_engine.query(query_text)

    num_source_nodes = len(response.source_nodes)
    print(f"Number of source nodes: {num_source_nodes}")

    for s in response.source_nodes:
        print(f"Node Score: {s.score}")
        print(s.node)

    print(response.get_formatted_sources(length=2000))
    print(response)


if __name__ == "__main__":
    main()
