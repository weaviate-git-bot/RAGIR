import pandas as pd
import asyncio

from llama_index import set_global_service_context
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.evaluation import RetrieverEvaluator
from llama_index.evaluation.retrieval.base import RetrievalEvalMode

from RAGIR.database.setup_database import WeaviateVectorDB
from RAGIR.utils.util_functions import (
    initialize_embedding,
    initialize_llama_cpp,
    initialize_service_context
)


class Evaluate:

    def __init__(self, retriever_file_path, retriever, metric_names) -> None:
        self.retriever_file_path = retriever_file_path
        self.retriever = retriever
        self.metric_names = metric_names
        self.qa_dataset = self._read_retriever(
            retriever_file_path=self.retriever_file_path
        )

    def _read_retriever(self, retriever_file_path):
        return EmbeddingQAFinetuneDataset.from_json(retriever_file_path)

    async def aevaluate_retriever_dataset(self, name):
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            metric_names=self.metric_names, retriever=self.retriever
        )
        mode = RetrievalEvalMode.from_str(self.qa_dataset.mode)
        for query_id, query in self.qa_dataset.queries.items():
            print(query_id, query)

        eval_results = await retriever_evaluator.aevaluate_dataset(
            self.qa_dataset,
            show_progress=True
        )

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)
        print(metric_dicts)
        full_df = pd.DataFrame(metric_dicts)
        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        metric_df = pd.DataFrame(
            {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
        )

        metric_df.to_json("results.json")


def main():
    retriever_file_path = "/Users/torky/Documents/thesis-torky/retriever_pairs.json"
    model_path = "/Users/torky/Documents/LlamaCPP-2/zephyr-7b-beta.Q4_0.gguf"
    embedding_model_name = "BAAI/bge-small-en-v1.5"

    embed_model = initialize_embedding(model_name=embedding_model_name)
    llm = initialize_llama_cpp(
        model_path=model_path,
        temperature=0,
        max_new_tokens=1024,
        context_window=3900,
        model_kwargs={"n_gpu_layers": 2},
        verbose=False,
    )

    service_context = initialize_service_context(
        embed_model=embed_model, llm=llm
    )
    set_global_service_context(service_context)

    weaviateVectorDB = WeaviateVectorDB(
        client_url="http://localhost:8080",
        embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
        index_name="Test",
        documents_path="/Users/torky/Documents/thesis-torky/docs/test",
    )
    retriever = weaviateVectorDB.index.as_retriever()

    evaluate = Evaluate(
        retriever_file_path=retriever_file_path,
        retriever=retriever,
        metric_names=["mrr", "hit_rate"]
        )

    async def evaluate_retriever(evaluate, name):
        await evaluate.aevaluate_retriever_dataset(name=name)

    asyncio.run(evaluate_retriever(evaluate, name="top-2 eval"))

if __name__ == "__main__":
    main()
