from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from config import HISTORY_MESSAGES, MODEL_NAME, RETRIEVAL_K
from models import get_llm
from prompts import ANSWER_PROMPT, REWRITE_PROMPT


def format_docs(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )


def get_chat_history(messages):
    history = []

    for message in messages[-HISTORY_MESSAGES:]:
        if message["role"] == "user":
            history.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            history.append(
                AIMessage(content=message["content"])
            )

    return history


def build_chains(vector_store):
    model = get_llm(MODEL_NAME)
    parser = StrOutputParser()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    rewrite_chain = REWRITE_PROMPT | model | parser

    parallel_chain = RunnableParallel(
        {
            "context": (
                RunnableLambda(lambda data: data["search_query"])
                | retriever
                | RunnableLambda(format_docs)
            ),
            "question": RunnableLambda(
                lambda data: data["question"]
            ),
            "history": RunnableLambda(
                lambda data: data["history"]
            ),
        }
    )

    answer_chain = (
        parallel_chain
        | ANSWER_PROMPT
        | model
        | parser
    )

    return rewrite_chain, answer_chain


def prepare_search_query(
    question,
    history,
    rewrite_chain,
    use_followup_context,
):
    if history and use_followup_context:
        rewritten_question = rewrite_chain.invoke(
            {
                "history": history,
                "question": question,
            }
        ).strip()

        return rewritten_question or question

    return question