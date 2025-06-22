from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from generate_embeddings import generate_embeddings
from generator import init_llm
from parsing_data import parsing_docs_simpson_archive
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


def create_retriever(url, k):
    docs = parsing_docs_simpson_archive(url)
    embeddings = generate_embeddings(docs)
    retriever = init_retriever(embeddings, k)

    return retriever


def build_url(urls, temperature=1, max_tokens=256, top_p=1, max_retries=3, k=3):
    retriever = create_retriever(urls, k)
    llm, llm_prompt = init_llm(temperature, max_tokens, top_p, max_retries)
    rag_chain = create_rag_chain(retriever, llm, llm_prompt)

    return rag_chain


def build(retriever, temperature=1, max_tokens=256, top_p=1, max_retries=3):
    llm, llm_prompt = init_llm(temperature, max_tokens, top_p, max_retries)
    rag_chain = create_rag_chain(retriever, llm, llm_prompt)

    return rag_chain
