from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from generate_embeddings import generate_embeddings
from generator import init_llm
from parsing_data import parsing_docs_simpson_park
from retriever import init_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm, llm_prompt):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def create_retriever(url):
    docs = parsing_docs_simpson_park(url)
    embeddings = generate_embeddings(docs)
    retriever = init_retriever(embeddings)

    return retriever
def build(retriever):
    llm, llm_prompt = init_llm()
    rag_chain = create_rag_chain(retriever, llm, llm_prompt)

    return rag_chain
