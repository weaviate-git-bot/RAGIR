import re
import pickle
import pandas as pd
import json

from llama_index.schema import BaseNode
from llama_index.prompts import (
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)
from typing import Tuple, List
from RAGIR.utils.util_functions import (
    initialize_embedding,
    initialize_llama_cpp,
)
from RAGIR.database.setup_database import WeaviateVectorDB
from llama_index.evaluation import generate_question_context_pairs


QUESTION_GEN_USER_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate the relevant questions. "
)

QUESTION_GEN_SYS_TMPL = """\
You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.\
"""

QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)


class GenerateData:
    def __init__(self, llm, nodes) -> None:
        self.llm = llm
        self.nodes = nodes

    def _generate_question_gen_template(self):
        question_gen_template = ChatPromptTemplate(
            message_templates=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=QUESTION_GEN_SYS_TMPL
                    ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=QUESTION_GEN_USER_TMPL
                    ),
            ]
        )
        return question_gen_template

    def _generate_answers_for_questions(
        self, questions: List[str], context: str, llm
    ) -> str:
        """Generate answers for questions given context."""
        answers = []
        for question in questions:
            fmt_qa_prompt = QA_PROMPT.format(
                context_str=context,
                query_str=question
                )
            response_obj = llm.complete(fmt_qa_prompt)
            answers.append(str(response_obj))
        return answers

    def _generate_qa_pairs(
        self, nodes: List[BaseNode], llm, num_questions_per_chunk: int = 1
    ) -> List[Tuple[str, str]]:
        """Generate questions."""
        qa_pairs = []
        question_gen_template = self._generate_question_gen_template()
        for idx, node in enumerate(nodes):
            print(f"Node {idx}/{len(nodes)}")
            context_str = node.get_content(metadata_mode="all")
            fmt_messages = question_gen_template.format_messages(
                num_questions_per_chunk=num_questions_per_chunk,
                context_str=context_str,
            )
            chat_response = llm.chat(fmt_messages)
            raw_output = chat_response.message.content
            result_list = str(raw_output).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result_list
            ]
            answers = self._generate_answers_for_questions(
                cleaned_questions, context_str, llm
            )
            cur_qa_pairs = list(zip(cleaned_questions, answers))
            qa_pairs.extend(cur_qa_pairs)
        return qa_pairs

    def _generate_retriever_pairs(self):
        return generate_question_context_pairs(
            nodes=self.nodes,
            llm=self.llm,
            num_questions_per_chunk=1
        )

    def create_qa_pairs(self, num_questions_per_chunk):
        qa_pairs = self._generate_qa_pairs(
            self.nodes,
            self.llm,
            num_questions_per_chunk=num_questions_per_chunk,
        )
        return qa_pairs

    def create_retriever_pairs(self):
        return self._generate_retriever_pairs()

    def save_qa_pairs(self, qa_pairs, file_name="eval_dataset.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(qa_pairs, f)

    def save_retriever_pairs(self, retriever_pairs, file_name="retriever_pairs.json"):
        retriever_pairs.save_json(file_name)


def main():

    client_url = "http://localhost:8080"
    embedding_model = "BAAI/bge-small-en-v1.5"
    model_path = "/Users/torky/Documents/LlamaCPP-2/zephyr-7b-beta.Q4_0.gguf"
    documents_path = "/Users/torky/Documents/thesis-torky/docs/test"
    num_questions_per_chunk = 1
    index_name = "Test"

    weaviate_db = WeaviateVectorDB(
        client_url=client_url,
        embedding_model=embedding_model,
        index_name=index_name,
        documents_path=documents_path,
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

    generate_data = GenerateData(
        llm=llm,
        nodes=weaviate_db.nodes
        )

    # qa_pairs = generaxte_data.create_qa_pairs(
    #     num_questions_per_chunk=num_questions_per_chunk
    #     )
    # generate_data.save_qa_pairs(qa_pairs)

    retriever_pairs = generate_data.create_retriever_pairs()
    generate_data.save_retriever_pairs(retriever_pairs)


if __name__ == "__main__":
    main()
